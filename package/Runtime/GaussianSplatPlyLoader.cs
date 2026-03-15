// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace GaussianSplatting.Runtime
{
    public struct InputSplatData
    {
        public Vector3 pos;
        public Vector3 nor;
        public Vector3 dc0;
        public Vector3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
        public float opacity;
        public Vector3 scale;
        public Quaternion rot;
    }

    public static class GaussianSplatPlyLoader
    {
        enum ElementType { None, Float, Double, UChar }

        static int TypeToSize(ElementType t) => t switch
        {
            ElementType.Float => 4,
            ElementType.Double => 8,
            ElementType.UChar => 1,
            _ => 0
        };

        /// <summary>
        /// Parse a binary little-endian PLY from raw bytes and load it into the given
        /// <see cref="GaussianSplatRenderer"/> at runtime. Uses VeryHigh quality (Float32)
        /// format for all data — no quantization or clustering.
        /// </summary>
        /// <param name="renderer">Target renderer (must have shaders/compute assigned).</param>
        /// <param name="plyBytes">Raw PLY file bytes.</param>
        /// <param name="colmapToUnity">If true, converts from COLMAP (right-handed, Y-down) to Unity (left-handed, Y-up).</param>
        public static void LoadFromPlyBytes(GaussianSplatRenderer renderer, byte[] plyBytes, bool colmapToUnity = false)
        {
            ParsePlyHeader(plyBytes, out int vertexCount, out int vertexStride, out var attrs, out int dataStart);
            if (vertexCount <= 0)
                throw new InvalidOperationException("PLY has no vertices");

            var rawData = new NativeArray<byte>(vertexCount * vertexStride, Allocator.TempJob);
            NativeArray<byte>.Copy(plyBytes, dataStart, rawData, 0, vertexCount * vertexStride);

            NativeArray<InputSplatData> splats;
            unsafe
            {
                splats = PlyDataToSplats(rawData, vertexCount, vertexStride, attrs);
                ReorderSHs(vertexCount, (float*)splats.GetUnsafePtr());
                LinearizeData(splats);
            }
            rawData.Dispose();

            if (colmapToUnity)
                ConvertColmapToUnity(splats);

            var posData = BuildPositionData(splats);
            var otherData = BuildOtherData(splats);
            var (colorData, texWidth, texHeight) = BuildColorData(splats);
            var shData = BuildSHData(splats);

            Vector3 boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < vertexCount; i++)
            {
                boundsMin = Vector3.Min(boundsMin, splats[i].pos);
                boundsMax = Vector3.Max(boundsMax, splats[i].pos);
            }

            splats.Dispose();

            renderer.SetRuntimeSplatData(
                vertexCount,
                posData, otherData, colorData, shData,
                texWidth, texHeight,
                GaussianSplatAsset.VectorFormat.Float32,
                GaussianSplatAsset.VectorFormat.Float32,
                GaussianSplatAsset.SHFormat.Float32,
                GaussianSplatAsset.ColorFormat.Float32x4,
                boundsMin, boundsMax
            );

            posData.Dispose();
            otherData.Dispose();
            colorData.Dispose();
            shData.Dispose();

            Debug.Log($"[GaussianSplatPlyLoader] Loaded {vertexCount} splats from PLY ({plyBytes.Length / (1024f * 1024f):F1}MB)");
        }

        #region PLY Header Parsing

        static void ParsePlyHeader(byte[] data, out int vertexCount, out int vertexStride,
            out List<(string name, ElementType type)> attrs, out int dataStart)
        {
            vertexCount = 0;
            vertexStride = 0;
            attrs = new List<(string, ElementType)>();
            bool gotBinaryLE = false;
            int pos = 0;

            for (int lineIdx = 0; lineIdx < 9000; lineIdx++)
            {
                string line = ReadLine(data, ref pos);
                if (line == "end_header" || line.Length == 0)
                    break;
                var tokens = line.Split(' ');
                if (tokens.Length == 3 && tokens[0] == "format" && tokens[1] == "binary_little_endian")
                    gotBinaryLE = true;
                if (tokens.Length == 3 && tokens[0] == "element" && tokens[1] == "vertex")
                    vertexCount = int.Parse(tokens[2]);
                if (tokens.Length == 3 && tokens[0] == "property")
                {
                    ElementType type = tokens[1] switch
                    {
                        "float" => ElementType.Float,
                        "double" => ElementType.Double,
                        "uchar" => ElementType.UChar,
                        _ => ElementType.None
                    };
                    vertexStride += TypeToSize(type);
                    attrs.Add((tokens[2], type));
                }
            }

            if (!gotBinaryLE)
                throw new InvalidOperationException("PLY must be binary little-endian format");

            dataStart = pos;
        }

        static string ReadLine(byte[] data, ref int pos)
        {
            var sb = new StringBuilder();
            while (pos < data.Length)
            {
                byte b = data[pos++];
                if (b == '\n') break;
                sb.Append((char)b);
            }
            if (sb.Length > 0 && sb[sb.Length - 1] == '\r')
                sb.Remove(sb.Length - 1, 1);
            return sb.ToString();
        }

        #endregion

        #region PLY → InputSplatData

        static readonly string[] SplatAttributes =
        {
            "x", "y", "z", "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3", "f_rest_4",
            "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_8", "f_rest_9",
            "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14",
            "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19",
            "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23", "f_rest_24",
            "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29",
            "f_rest_30", "f_rest_31", "f_rest_32", "f_rest_33", "f_rest_34",
            "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39",
            "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        };

        static unsafe NativeArray<InputSplatData> PlyDataToSplats(
            NativeArray<byte> input, int count, int stride,
            List<(string name, ElementType type)> attributes)
        {
            var fileAttrOffsets = new NativeArray<int>(attributes.Count, Allocator.Temp);
            int offset = 0;
            for (int i = 0; i < attributes.Count; i++)
            {
                fileAttrOffsets[i] = offset;
                offset += TypeToSize(attributes[i].type);
            }

            var srcOffsets = new NativeArray<int>(SplatAttributes.Length, Allocator.Temp);
            for (int ai = 0; ai < SplatAttributes.Length; ai++)
            {
                int attrIndex = -1;
                for (int j = 0; j < attributes.Count; j++)
                {
                    if (attributes[j].name == SplatAttributes[ai] && attributes[j].type == ElementType.Float)
                    {
                        attrIndex = j;
                        break;
                    }
                }
                srcOffsets[ai] = attrIndex >= 0 ? fileAttrOffsets[attrIndex] : -1;
            }

            var dst = new NativeArray<InputSplatData>(count, Allocator.TempJob);
            int dstStride = UnsafeUtility.SizeOf<InputSplatData>();

            byte* srcPtr = (byte*)input.GetUnsafeReadOnlyPtr();
            byte* dstPtr = (byte*)dst.GetUnsafePtr();
            for (int i = 0; i < count; i++)
            {
                for (int attr = 0; attr < dstStride / 4; attr++)
                {
                    if (srcOffsets[attr] >= 0)
                        *(int*)(dstPtr + attr * 4) = *(int*)(srcPtr + srcOffsets[attr]);
                }
                srcPtr += stride;
                dstPtr += dstStride;
            }

            fileAttrOffsets.Dispose();
            srcOffsets.Dispose();
            return dst;
        }

        [BurstCompile]
        static unsafe void ReorderSHs(int splatCount, float* data)
        {
            int splatStride = UnsafeUtility.SizeOf<InputSplatData>() / 4;
            int shStartOffset = 9, shCount = 15;
            float* tmp = stackalloc float[shCount * 3];
            int idx = shStartOffset;
            for (int i = 0; i < splatCount; i++)
            {
                for (int j = 0; j < shCount; j++)
                {
                    tmp[j * 3 + 0] = data[idx + j];
                    tmp[j * 3 + 1] = data[idx + j + shCount];
                    tmp[j * 3 + 2] = data[idx + j + shCount * 2];
                }
                for (int j = 0; j < shCount * 3; j++)
                    data[idx + j] = tmp[j];
                idx += splatStride;
            }
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                splat.scale = GaussianUtils.LinearScale(splat.scale);
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            new LinearizeDataJob { splatData = splatData }
                .Schedule(splatData.Length, 4096).Complete();
        }

        #endregion

        #region COLMAP → Unity Coordinate Conversion

        [BurstCompile]
        struct ConvertColmapJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;

            public void Execute(int index)
            {
                var splat = splatData[index];
                // COLMAP Y-down → Unity Y-up: negate Y position
                splat.pos.y = -splat.pos.y;
                splatData[index] = splat;
            }
        }

        static void ConvertColmapToUnity(NativeArray<InputSplatData> splatData)
        {
            new ConvertColmapJob { splatData = splatData }
                .Schedule(splatData.Length, 4096).Complete();
        }

        #endregion

        #region Build UGS Binary Data

        static NativeArray<byte> BuildPositionData(NativeArray<InputSplatData> splats)
        {
            int stride = 12; // 3 × float
            var data = new NativeArray<byte>(splats.Length * stride, Allocator.TempJob);
            unsafe
            {
                byte* ptr = (byte*)data.GetUnsafePtr();
                for (int i = 0; i < splats.Length; i++)
                {
                    var p = splats[i].pos;
                    *(float*)(ptr) = p.x;
                    *(float*)(ptr + 4) = p.y;
                    *(float*)(ptr + 8) = p.z;
                    ptr += stride;
                }
            }
            return data;
        }

        static uint EncodeQuatToNorm10(float4 v)
        {
            return (uint)(v.x * 1023.5f) |
                   ((uint)(v.y * 1023.5f) << 10) |
                   ((uint)(v.z * 1023.5f) << 20) |
                   ((uint)(v.w * 3.5f) << 30);
        }

        static NativeArray<byte> BuildOtherData(NativeArray<InputSplatData> splats)
        {
            int stride = 16; // 4 bytes rot + 12 bytes scale (Float32)
            var data = new NativeArray<byte>(splats.Length * stride, Allocator.TempJob);
            unsafe
            {
                byte* ptr = (byte*)data.GetUnsafePtr();
                for (int i = 0; i < splats.Length; i++)
                {
                    var rot = splats[i].rot;
                    uint enc = EncodeQuatToNorm10(new float4(rot.x, rot.y, rot.z, rot.w));
                    *(uint*)ptr = enc;

                    var s = splats[i].scale;
                    *(float*)(ptr + 4) = s.x;
                    *(float*)(ptr + 8) = s.y;
                    *(float*)(ptr + 12) = s.z;
                    ptr += stride;
                }
            }
            return data;
        }

        static int SplatIndexToTextureIndex(uint idx)
        {
            uint2 xy = GaussianUtils.DecodeMorton2D_16x16(idx);
            uint width = GaussianSplatAsset.kTextureWidth / 16;
            idx >>= 8;
            uint x = (idx % width) * 16 + xy.x;
            uint y = (idx / width) * 16 + xy.y;
            return (int)(y * GaussianSplatAsset.kTextureWidth + x);
        }

        static (NativeArray<byte> data, int width, int height) BuildColorData(NativeArray<InputSplatData> splats)
        {
            var (width, height) = GaussianSplatAsset.CalcTextureSize(splats.Length);
            int pixelCount = width * height;
            var data = new NativeArray<byte>(pixelCount * 16, Allocator.TempJob); // Float32x4 = 16 bytes/pixel
            unsafe
            {
                float* fPtr = (float*)data.GetUnsafePtr();
                // zero-init is handled by NativeArray
                for (int i = 0; i < splats.Length; i++)
                {
                    int texIdx = SplatIndexToTextureIndex((uint)i);
                    fPtr[texIdx * 4 + 0] = splats[i].dc0.x;
                    fPtr[texIdx * 4 + 1] = splats[i].dc0.y;
                    fPtr[texIdx * 4 + 2] = splats[i].dc0.z;
                    fPtr[texIdx * 4 + 3] = splats[i].opacity;
                }
            }
            return (data, width, height);
        }

        static NativeArray<byte> BuildSHData(NativeArray<InputSplatData> splats)
        {
            int stride = UnsafeUtility.SizeOf<GaussianSplatAsset.SHTableItemFloat32>();
            var data = new NativeArray<byte>(splats.Length * stride, Allocator.TempJob);
            unsafe
            {
                var ptr = (GaussianSplatAsset.SHTableItemFloat32*)data.GetUnsafePtr();
                for (int i = 0; i < splats.Length; i++)
                {
                    var s = splats[i];
                    ptr[i] = new GaussianSplatAsset.SHTableItemFloat32
                    {
                        sh1 = s.sh1, sh2 = s.sh2, sh3 = s.sh3, sh4 = s.sh4,
                        sh5 = s.sh5, sh6 = s.sh6, sh7 = s.sh7, sh8 = s.sh8,
                        sh9 = s.sh9, shA = s.shA, shB = s.shB, shC = s.shC,
                        shD = s.shD, shE = s.shE, shF = s.shF,
                        shPadding = default
                    };
                }
            }
            return data;
        }

        #endregion
    }
}
