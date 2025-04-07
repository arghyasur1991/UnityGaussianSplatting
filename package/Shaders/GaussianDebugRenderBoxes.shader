// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Debug/Render Boxes"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Front

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc
// Use ONE of the stereo rendering methods but not both simultaneously
#pragma multi_compile_local _ STEREO_MULTIVIEW_ON

// Ensure both stereo modes aren't active simultaneously
// Instead of error, prioritize multiview over instancing when both are defined
#if defined(STEREO_MULTIVIEW_ON) && defined(UNITY_STEREO_INSTANCING_ENABLED)
    // When both are defined, prioritize STEREO_MULTIVIEW_ON and disable instancing
    #undef UNITY_STEREO_INSTANCING_ENABLED
    #undef STEREO_INSTANCING_ON
    #undef UNITY_SINGLE_PASS_STEREO
    // Keep multiview features enabled, making it the only active stereo mode
    #define ONLY_USE_MULTIVIEW
#endif

// Enable instancing modes if multiview is not defined
#if !defined(STEREO_MULTIVIEW_ON)
    #pragma multi_compile_local _ UNITY_STEREO_INSTANCING_ENABLED UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON
#endif

// Include UnityCG.cginc first for stereo rendering macros
#include "UnityCG.cginc"
#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;

bool _DisplayChunks;

// Multiview is already handled in GaussianSplatting.hlsl
// No need to declare UNITY_DECLARE_MULTIVIEW here to avoid gl_ViewID redefinition

struct v2f
{
    half4 col : COLOR0;
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

float _SplatScale;
float _SplatOpacityScale;

// based on https://iquilezles.org/articles/palettes/
// cosine based palette, 4 vec3 params
half3 palette(float t, half3 a, half3 b, half3 c, half3 d)
{
    return a + b*cos(6.28318*(c*t+d));
}

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o;
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    // We're using SV_InstanceID directly, no need for UNITY_SETUP_INSTANCE_ID
    
    bool chunks = _DisplayChunks;
	uint idx = vtxID;
	float3 localPos = float3(idx&1, (idx>>1)&1, (idx>>2)&1) * 2.0 - 1.0;

    float3 centerWorldPos = 0;

    if (!chunks)
    {
        // display splat boxes
        instID = _OrderBuffer[instID];
        SplatData splat = LoadSplatData(instID);

        float4 boxRot = splat.rot;
        float3 boxSize = splat.scale;
        boxSize *= _SplatScale;

        float3x3 splatRotScaleMat = CalcMatrixFromRotationScale(boxRot, boxSize);
        splatRotScaleMat = mul((float3x3)unity_ObjectToWorld, splatRotScaleMat);

        centerWorldPos = splat.pos;
        centerWorldPos = mul(unity_ObjectToWorld, float4(centerWorldPos,1)).xyz;

        o.col.rgb = saturate(splat.sh.col);
        o.col.a = saturate(splat.opacity * _SplatOpacityScale);

        localPos = mul(splatRotScaleMat, localPos) * 2;
    }
    else
    {
        // display chunk boxes
        localPos = localPos * 0.5 + 0.5;
        SplatChunkInfo chunk = _SplatChunks[instID];
        float3 posMin = float3(chunk.posX.x, chunk.posY.x, chunk.posZ.x);
        float3 posMax = float3(chunk.posX.y, chunk.posY.y, chunk.posZ.y);

        localPos = lerp(posMin, posMax, localPos);
        localPos = mul(unity_ObjectToWorld, float4(localPos,1)).xyz;

        o.col.rgb = palette((float)instID / (float)_SplatChunkCount, half3(0.5,0.5,0.5), half3(0.5,0.5,0.5), half3(1,1,1), half3(0.0, 0.33, 0.67));
        o.col.a = 0.1;
    }

    float3 worldPos = centerWorldPos + localPos;
    
    o.vertex = UnityWorldToClipPos(worldPos);
    
    FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

half4 frag (v2f i) : SV_Target
{
    #if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(STEREO_MULTIVIEW_ON) || defined(UNITY_SINGLE_PASS_STEREO)
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    #endif
    
    half4 res = half4(i.col.rgb * i.col.a, i.col.a);
    return res;
}
ENDCG
        }
    }
}
