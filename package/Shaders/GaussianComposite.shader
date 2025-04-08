// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/Composite"
{
    Properties
    {
        _IsStereoEnabled("Stereo Enabled", Int) = 0
    }
    
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

// Use ONE of the stereo rendering methods but not both simultaneously
#pragma multi_compile_local _ STEREO_MULTIVIEW_ON

// Prevent both STEREO_MULTIVIEW_ON and UNITY_STEREO_INSTANCING_ENABLED from being active together
#if defined(STEREO_MULTIVIEW_ON)
    // If multiview is enabled, explicitly DISABLE instancing
    #define DISABLE_STEREO_INSTANCING
    #undef UNITY_STEREO_INSTANCING_ENABLED
    #undef STEREO_INSTANCING_ON
    #undef UNITY_SINGLE_PASS_STEREO
#else
    // Only enable instancing if multiview is NOT enabled
    #pragma multi_compile_local _ UNITY_STEREO_INSTANCING_ENABLED UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON
#endif

#include "UnityCG.cginc"

struct v2f
{
    float4 vertex : SV_POSITION;
    float2 uv : TEXCOORD0;
    UNITY_VERTEX_OUTPUT_STEREO
};

struct appdata
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
    uint vtxID : SV_VertexID;
    UNITY_VERTEX_INPUT_INSTANCE_ID //Insert
};

float4 _GaussianSplatRT_TexelSize;
#ifndef _ISSTENABLED_DEFINED
#define _ISSTENABLED_DEFINED
uint _IsStereoEnabled;
#endif


UNITY_DECLARE_SCREENSPACE_TEXTURE(_GaussianSplatRT);
// Texture2D _GaussianSplatRT_t;
// SamplerState _GaussianSplatRT_s;


v2f vert (appdata v)
{
    v2f o;
    UNITY_SETUP_INSTANCE_ID(v); 
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

    uint vtxID = v.vtxID;

    // o.vertex = UnityObjectToClipPos(v.vertex);

    // o.uv = v.uv;
    
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = UnityObjectToClipPos(float4(quadPos, 1, 1));
    // o.vertex = UnityObjectToClipPos(float4(quadPos, 1, 1));

    o.uv = float2(vtxID&1, (vtxID>>1)&1);
    
    return o;
}

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    
    half4 col;
    // Normalize the pixel coordinates to [0,1] range
    float2 normalizedUV = float2(i.vertex.x / _ScreenParams.x, i.vertex.y / _ScreenParams.y);
    int3 uv = int3(int(i.vertex.x), int(i.vertex.y), 0);
    
    col = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_GaussianSplatRT, normalizedUV);
    // col =  _GaussianSplatRT_t.Sample(_GaussianSplatRT_s, i.uv);
    // col =  _GaussianSplatRT_t.Sample(_GaussianSplatRT_s, uint2(int3(int(i.vertex.x), int(i.vertex.y), 0).xy));

    // col =  _GaussianSplatRT_t.Load((int3(int(i.vertex.x), int(i.vertex.y), 0)));
    // col = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_GaussianSplatRT, i.uv);

    // col = _GaussianSplatRT.Load(int3(i.vertex.xy, 0));
    // col = tex2D(_GaussianSplatRT, int3(i.vertex.xy, 0));// _GaussianSplatRT.Load(int3(i.vertex.xy, 0));
    
    col.rgb = GammaToLinearSpace(col.rgb);
    col.a = saturate(col.a * 1.5);
    return col;
}
ENDCG
        }
    }
}
