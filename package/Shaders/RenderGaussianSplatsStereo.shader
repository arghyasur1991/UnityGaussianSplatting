// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats Stereo"
{
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
#pragma geometry geom
#pragma fragment frag
#pragma require compute
#pragma require 2darray
#pragma target 4.5

// Only compile this pass for stereo modes
#pragma multi_compile_local _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

#include "GaussianSplatting.hlsl"
#include "UnityCG.cginc"

StructuredBuffer<uint> _OrderBuffer;

struct appdata
{
    uint vtxID : SV_VertexID;
    uint instID : SV_InstanceID;
};

struct v2g
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
    uint instID : TEXCOORD1;
    UNITY_VERTEX_OUTPUT_STEREO
};

struct g2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
    uint rtIndex : SV_RenderTargetArrayIndex;
    UNITY_VERTEX_OUTPUT_STEREO
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;

v2g vert (appdata v)
{
    v2g o = (v2g)0;
    UNITY_INITIALIZE_OUTPUT(v2g, o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    
    uint vtxID = v.vtxID;
    uint instID = v.instID / 2;
    instID = _OrderBuffer[instID];
    o.instID = v.instID;
    
    // Get the eye index to determine which view data to use
    uint eyeIndex = v.instID & 1;
    uint viewIndex = instID * 2 + eyeIndex;
    
    SplatViewData view = _SplatViewData[viewIndex];
    float4 centerClipPos = view.pos;
    bool behindCam = centerClipPos.w <= 0;
    if (behindCam)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
    }
    else
    {
        o.col.r = f16tof32(view.color.x >> 16);
        o.col.g = f16tof32(view.color.x);
        o.col.b = f16tof32(view.color.y >> 16);
        o.col.a = f16tof32(view.color.y);

        uint idx = vtxID;
        float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
        quadPos *= 2;

        o.pos = quadPos;

        float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2 / _ScreenParams.xy;
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

// Geometry shader that passes through the triangle and sets the render target array index
[maxvertexcount(3)]
void geom(triangle v2g input[3], inout TriangleStream<g2f> outStream)
{
    // Don't process if any vertex is NaN (culled behind camera)
    if (any(isnan(input[0].vertex)) || any(isnan(input[1].vertex)) || any(isnan(input[2].vertex)))
        return;
    
    g2f o;
    UNITY_INITIALIZE_OUTPUT(g2f, o);
    UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(input[0], o);
    
    // Set the render target array index based on the eye
    o.rtIndex = input[0].instID & 1;
    
    // Pass through all three vertices of the triangle
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        o.vertex = input[i].vertex;
        o.col = input[i].col;
        o.pos = input[i].pos;
        outStream.Append(o);
    }
    outStream.RestartStrip();
}

half4 frag (g2f i) : SV_Target
{
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