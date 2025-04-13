// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/CompositeStereo"
{
    SubShader
    {
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #pragma require compute
            #pragma require 2darray
            #pragma target 4.5

            // Only compile this pass for stereo modes
            #pragma multi_compile_local _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

            #include "UnityCG.cginc"

            struct appdata
            {
                uint vtxID : SV_VertexID;
                uint instanceID : SV_InstanceID;
            };

            struct v2g
            {
                float4 vertex : SV_POSITION;
                uint instanceID : SV_InstanceID;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            struct g2f
            {
                float4 vertex : SV_POSITION;
                uint rtIndex : SV_RenderTargetArrayIndex;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2g vert(appdata v)
            {
                v2g o;
                UNITY_INITIALIZE_OUTPUT(v2g, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                
                uint vtxID = v.vtxID;
                float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
                o.vertex = UnityObjectToClipPos(float4(quadPos, 1, 1));
                o.instanceID = v.instanceID;
                return o;
            }

            // Geometry shader that passes through the triangle and sets the render target array index
            [maxvertexcount(3)]
            void geom(triangle v2g input[3], inout TriangleStream<g2f> outStream)
            {
                g2f o;
                UNITY_INITIALIZE_OUTPUT(g2f, o);
                UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(input[0], o);
                
                // Use the instanceID to set which eye/slice to render to
                o.rtIndex = input[0].instanceID;
                
                // Pass through all three vertices of the triangle
                [unroll]
                for (int i = 0; i < 3; i++)
                {
                    o.vertex = input[i].vertex;
                    outStream.Append(o);
                }
                outStream.RestartStrip();
            }

            // Separate textures for left and right eyes
            UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);

            half4 frag(g2f i) : SV_Target
            {
                half4 col;
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                
                // Normalize the pixel coordinates to [0,1] range
                float2 normalizedUV = float2(i.vertex.x / _ScreenParams.x, i.vertex.y / _ScreenParams.y);
                col = UNITY_SAMPLE_TEX2DARRAY(_GaussianSplatRT, float3(normalizedUV, i.rtIndex));

                col.rgb = GammaToLinearSpace(col.rgb);
                col.a = saturate(col.a * 1.5);
                return col;
            }
            ENDCG
        }
    }
}
