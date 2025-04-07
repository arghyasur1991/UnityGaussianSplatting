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

float4 _GaussianSplatRT_TexelSize;
#ifndef _ISSTENABLED_DEFINED
#define _ISSTENABLED_DEFINED
uint _IsStereoEnabled;
#endif

// Platform-specific texture declarations
#if defined(SHADER_API_MOBILE) || defined(SHADER_API_GLES3) || defined(SHADER_API_GLES) || defined(SHADER_API_VULKAN) || defined(SHADER_API_METAL)
// Mobile platforms use standard 2D textures
sampler2D _GaussianSplatRT;
#else
// Desktop platforms use array textures for stereo
UNITY_DECLARE_SCREENSPACE_TEXTURE(_GaussianSplatRT);
#if defined(STEREO_MULTIVIEW_ON) || defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON)
UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#endif
#endif

v2f vert (uint vtxID : SV_VertexID)
{
    v2f o;
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = float4(quadPos, 1, 1);
    o.uv = float2(vtxID&1, (vtxID>>1)&1);
    
    return o;
}

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    
    half4 col;
    
    // Handle stereo rendering with platform-specific sampling
    if (_IsStereoEnabled)
    {
        #if defined(SHADER_API_MOBILE) || defined(SHADER_API_GLES3) || defined(SHADER_API_GLES) || defined(SHADER_API_VULKAN) || defined(SHADER_API_METAL)
            // Mobile platforms - use standard texture sampling
            col = tex2D(_GaussianSplatRT, i.uv);
        #else
            // Desktop platforms - use array texture for stereo
            col = tex2DArray(_GaussianSplatRT, float3(i.uv, unity_StereoEyeIndex));
        #endif
    }
    else
    {
        // Standard non-stereo sampling
        #if defined(SHADER_API_MOBILE) || defined(SHADER_API_GLES3) || defined(SHADER_API_GLES) || defined(SHADER_API_VULKAN) || defined(SHADER_API_METAL)
            col = tex2D(_GaussianSplatRT, i.uv);
        #else
            col = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_GaussianSplatRT, i.uv);
        #endif
    }
    
    col.rgb = GammaToLinearSpace(col.rgb);
    col.a = saturate(col.a * 1.5);
    return col;
}
ENDCG
        }
    }
}

