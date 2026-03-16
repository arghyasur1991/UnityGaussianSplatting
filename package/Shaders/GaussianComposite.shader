// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/Composite"
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
#pragma fragment frag
#pragma require compute
#pragma use_dxc
#pragma require 2darray

// Enable proper multi-compile support for all stereo rendering modes
#pragma multi_compile_local _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

#include "UnityCG.cginc"

struct v2f
{
    float4 vertex : SV_POSITION;
};

struct appdata
{
    float4 vertex : POSITION;
    uint vtxID : SV_VertexID;
};

v2f vert (uint vtxID : SV_VertexID)
{
    v2f o;
    
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = float4(quadPos, 1, 1);
    return o;
}

// Separate textures for left and right eyes
#if defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON) || defined(STEREO_MULTIVIEW_ON)
UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#else
Texture2D _GaussianSplatRT;
SamplerState sampler_LinearClamp;
#endif

int _CustomStereoEyeIndex;
half4 frag (v2f i) : SV_Target
{
    half4 col;
    float2 uv = float2(i.vertex.x / _ScreenParams.x, i.vertex.y / _ScreenParams.y);
    #if defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON) || defined(STEREO_MULTIVIEW_ON)
        col = UNITY_SAMPLE_TEX2DARRAY(_GaussianSplatRT, float3(uv, _CustomStereoEyeIndex));
    #else
        col = _GaussianSplatRT.SampleLevel(sampler_LinearClamp, uv, 0);
    #endif

    return float4(GammaToLinearSpace(col.rgb / col.a), col.a);
}
ENDCG
        }
    }
}
