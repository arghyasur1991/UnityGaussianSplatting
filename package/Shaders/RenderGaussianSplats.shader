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
#if defined(SHADER_API_METAL)
sampler2D _GaussianSplatRT;
#elif defined(STEREO_MULTIVIEW_ON) || defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON)
// Only declare array texture for any stereo mode
UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#else
// Standard texture for non-stereo mode
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
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    instID = _OrderBuffer[instID];
    SplatViewData view = _SplatViewData[instID];
    
    // Get the appropriate position for the current eye
    float4 centerClipPos = view.pos;
    
    uint eyeIndex = 0;
    // Safe way to get eye index without trying to modify gl_ViewID
    #if defined(STEREO_MULTIVIEW_ON)
    eyeIndex = unity_StereoEyeIndex;
    #elif defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON)
    eyeIndex = unity_StereoEyeIndex;
    #endif
    
    bool behindCam = centerClipPos.w <= 0;
    if (behindCam)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
    }
    else
    {
        // Extract color data
        o.col.r = f16tof32(view.color >> 16);
        o.col.g = f16tof32(view.color);
        o.col.b = f16tof32(view.color2 >> 16);
        o.col.a = f16tof32(view.color2);

        uint idx = vtxID;
        float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
        quadPos *= 2;

        o.pos = quadPos;

        // Use the appropriate axis data for the current eye
        float2 axis1, axis2;
        if (_IsStereoEnabled)
        {
            axis1 = GetEyeAxis1(view, eyeIndex);
            axis2 = GetEyeAxis2(view, eyeIndex);
        }
        else
        {
            axis1 = view.axis1.xy;
            axis2 = view.axis2.xy;
        }

        float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / _ScreenParams.xy;
        
        // For stereo rendering, adjust the center position based on eye index
        #if defined(STEREO_MULTIVIEW_ON)
        if (_IsStereoEnabled && eyeIndex == 1)
        {
            // Simply add stereo convergence offset for right eye
            centerClipPos.x += _StereoSeparation;
        }
        #elif defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON)
        if (_IsStereoEnabled && eyeIndex == 1)
        {
            // Need to convert to right eye position
            float4 rightEyePos = UnityObjectToClipPos(
                mul(unity_WorldToObject, 
                    mul(unity_StereoCameraToWorld[1], 
                        mul(unity_StereoCameraInvProjection[1], 
                            float4(0, 0, -1, 1)))));
                            
            centerClipPos = rightEyePos;
        }
        #endif
        
        o.vertex = centerClipPos;
        o.vertex.xy += deltaScreenPos * centerClipPos.w;

        // is this splat selected?
        if (_SplatBitsValid)
        {
            uint wordIdx = instID / 32;
            uint bitIdx = instID & 31;
            uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
            if (selVal & (1 << bitIdx))
            {
                o.col.a = -1;				
            }
        }
    }
    FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

half4 frag (v2f i) : SV_Target
{
    // Safely setup stereo rendering, handles all VR rendering paths
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    if (i.col.a >= 0)
    {
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        // "selected" splat: magenta outline, increase opacity, magenta tint
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
