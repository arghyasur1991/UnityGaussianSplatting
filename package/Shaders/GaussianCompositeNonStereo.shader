// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/CompositeNonStereo"
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
            #pragma vertex vert_simple
            #pragma fragment frag_simple
            #pragma require compute
            #pragma use_dxc
            #pragma target 4.5

            #include "UnityCG.cginc"

            struct v2f_simple
            {
                float4 vertex : SV_POSITION;
            };

            v2f_simple vert_simple(uint vtxID : SV_VertexID)
            {
                v2f_simple o;
                float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
                o.vertex = UnityObjectToClipPos(float4(quadPos, 1, 1));
                return o;
            }

            Texture2D _GaussianSplatRT;

            half4 frag_simple(v2f_simple i) : SV_Target
            {
                half4 col = _GaussianSplatRT.Load(int3(i.vertex.xy, 0));
                col.rgb = GammaToLinearSpace(col.rgb);
                col.a = saturate(col.a * 1.5);
                return col;
            }
            ENDCG
        }
    }
}
