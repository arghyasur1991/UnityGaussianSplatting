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

// Enable proper multi-compile support for all stereo rendering modes
#pragma multi_compile_local _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

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
    UNITY_VERTEX_INPUT_INSTANCE_ID
};

v2f vert (appdata v)
{
    v2f o;
    UNITY_SETUP_INSTANCE_ID(v); 
    UNITY_INITIALIZE_OUTPUT(v2f, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

    uint vtxID = v.vtxID;
    
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = UnityObjectToClipPos(float4(quadPos, 1, 1));

    o.uv = float2(vtxID&1, (vtxID>>1)&1);
    return o;
}

// For backward compatibility
UNITY_DECLARE_SCREENSPACE_TEXTURE(_GaussianSplatRT);

// Separate textures for left and right eyes
Texture2D _LeftEyeTex;
// SAMPLER(sampler_LeftEyeTex);

Texture2D _RightEyeTex;
// SAMPLER(sampler_RightEyeTex);

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    
    // Normalize the pixel coordinates to [0,1] range
    float2 normalizedUV = float2(i.vertex.x / _ScreenParams.x, i.vertex.y / _ScreenParams.y);
    
    half4 col;
    
    // Check if using separate eye textures
    #if defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON) || defined(STEREO_MULTIVIEW_ON)
        if (unity_StereoEyeIndex == 0)
            col = _LeftEyeTex.Load(int3(i.vertex.xy, 0));
        else
            col = _RightEyeTex.Load(int3(i.vertex.xy, 0));
    #else
        // Fallback to legacy single-texture approach for backward compatibility
        col = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_GaussianSplatRT, normalizedUV);
    #endif
    
    col.rgb = GammaToLinearSpace(col.rgb);
    col.a = saturate(col.a * 1.5);
    return col;
}
ENDCG
        }
    }
}
