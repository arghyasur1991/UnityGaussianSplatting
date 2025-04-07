// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
    Properties
    {
        _IsStereoEnabled("Stereo Enabled", Int) = 0
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off
            
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
#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;
// Platform-specific texture declarations
// IMPORTANT: For Quest (multiview), we MUST use texture arrays even on mobile platforms
#if defined(STEREO_MULTIVIEW_ON)
    // When using multiview (Quest), we need texture arrays for stereo rendering
    UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#elif defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO)
    // For other stereo modes, use texture arrays as well
    UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#elif defined(SHADER_API_MOBILE) || defined(SHADER_API_GLES3) || defined(SHADER_API_GLES) || defined(SHADER_API_VULKAN) || defined(SHADER_API_METAL)
    // Mobile platforms use standard 2D textures when not in stereo mode
    sampler2D _GaussianSplatRT;
#else
    // Desktop platforms use screen space textures for non-stereo
    UNITY_DECLARE_SCREENSPACE_TEXTURE(_GaussianSplatRT);
#endif
float4 _GaussianSplatRT_TexelSize;
float _SplasPerBatch;
uint _SplatCount;
#ifndef _ISSTENABLED_DEFINED
#define _ISSTENABLED_DEFINED
uint _IsStereoEnabled;
#endif
float _StereoSeparation;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o;
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    
    // Set up stereo for VR
    #if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO)
        // For single-pass stereo, instance ID alternates between eyes
        if (_IsStereoEnabled > 0) 
        {
            unity_StereoEyeIndex = instID & 1;
            instID = instID >> 1; // Divide by 2 to get the actual splat index
        }
    #endif
    
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    // Get the correct splat from ordered buffer
    uint splatIdx = _OrderBuffer[instID]; 
    SplatViewData view = _SplatViewData[splatIdx];
    
    // Get the eye index for the correct view data
    uint eyeIndex = 0;
    #if defined(STEREO_MULTIVIEW_ON) || defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO)
        eyeIndex = unity_StereoEyeIndex;
    #endif
    
    // Get clip space position for current eye
    float4 centerClipPos = (eyeIndex == 0) ? view.pos : view.posRight;
    bool behindCam = centerClipPos.w <= 0;
    
    if (behindCam)
    {
        o.vertex = float4(0, 0, 0, 0); // Will be clipped
        o.col = half4(0, 0, 0, 0);
        o.pos = float2(0, 0);
        return o;
    }
    
    // Calculate quad corners
    uint idx = vtxID;
    float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
    quadPos *= 2.0; // Double size 
    o.pos = quadPos;
    
    // Get correct axes for current eye
    float2 axis1 = (eyeIndex == 0) ? view.axis1.xy : view.axis1.zw;
    float2 axis2 = (eyeIndex == 0) ? view.axis2.xy : view.axis2.zw;
    
    // Calculate screen space position delta
    float2 screenDelta = (quadPos.x * axis1 + quadPos.y * axis2) * 2.0 / _ScreenParams.xy;
    o.vertex = centerClipPos;
    o.vertex.xy += screenDelta * centerClipPos.w;
    
    // Extract color from packed values
    o.col.r = f16tof32(view.color >> 16);
    o.col.g = f16tof32(view.color);
    o.col.b = f16tof32(view.color2 >> 16);
    o.col.a = f16tof32(view.color2);
    
    // Handle selection highlighting
    if (_SplatBitsValid)
    {
        uint wordIdx = splatIdx / 32;
        uint bitIdx = splatIdx & 31;
        uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
        if (selVal & (1 << bitIdx))
        {
            o.col.a = -1; // Special value for selection
        }
    }
    
    // Use helper function from GaussianSplatting.hlsl for backbuffer flipping
    FlipProjectionIfBackbuffer(o.vertex);
    
    return o;
}

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    
    // Calculate gaussian alpha
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    
    if (i.col.a >= 0)
    {
        // Normal splat rendering
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        // "Selected" splat rendering
        half3 selectedColor = half3(1,0,1);
        if (alpha > 7.0/255.0)
        {
            if (alpha < 10.0/255.0)
            {
                alpha = 1;
                i.col.rgb = selectedColor;
            }
            alpha = saturate(alpha + 0.3);
        }
        i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
    }
    
    if (alpha < 1.0/255.0)
        discard;

    half4 res = half4(i.col.rgb * alpha, alpha);
    return res;
}
ENDCG
        }
    }
}
