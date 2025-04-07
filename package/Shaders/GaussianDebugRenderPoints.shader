// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Debug/Render Points"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite On
            Cull Off
            
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

// Multiview is already handled in GaussianSplatting.hlsl
// No need to declare UNITY_DECLARE_MULTIVIEW here to avoid gl_ViewID redefinition

struct v2f
{
    half3 color : TEXCOORD0;
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

float _SplatSize;
bool _DisplayIndex;
int _SplatCount;

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o;
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    // We're using SV_InstanceID directly, no need for UNITY_SETUP_INSTANCE_ID
    
    uint splatIndex = instID;

    SplatData splat = LoadSplatData(splatIndex);

    float3 centerWorldPos = splat.pos;
    centerWorldPos = mul(unity_ObjectToWorld, float4(centerWorldPos,1)).xyz;

    float4 centerClipPos = mul(UNITY_MATRIX_VP, float4(centerWorldPos, 1));

    o.vertex = centerClipPos;
	uint idx = vtxID;
    float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
    o.vertex.xy += (quadPos * _SplatSize / _ScreenParams.xy) * o.vertex.w;

    o.color.rgb = saturate(splat.sh.col);
    if (_DisplayIndex)
    {
        o.color.r = frac((float)splatIndex / (float)_SplatCount * 100);
        o.color.g = frac((float)splatIndex / (float)_SplatCount * 10);
        o.color.b = (float)splatIndex / (float)_SplatCount;
    }

    FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

half4 frag (v2f i) : SV_Target
{
    #if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(STEREO_MULTIVIEW_ON) || defined(UNITY_SINGLE_PASS_STEREO)
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    #endif
    
    return half4(i.color.rgb, 1);
}
ENDCG
        }
    }
}
