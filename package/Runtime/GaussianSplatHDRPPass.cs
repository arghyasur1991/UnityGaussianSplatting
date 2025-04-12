// SPDX-License-Identifier: MIT
#if GS_ENABLE_HDRP

using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    // Note: I have no idea what is the proper usage of CustomPass.
    // Code below "seems to work" but I'm just fumbling along, without understanding any of it.
    class GaussianSplatHDRPPass : CustomPass
    {
        RTHandle m_RenderTarget; // For backward compatibility
        RTHandle m_LeftEyeRT;
        RTHandle m_RightEyeRT;

        // It can be used to configure render targets and their clear state. Also to create temporary render target textures.
        // When empty this render pass will render to the active camera render target.
        // You should never call CommandBuffer.SetRenderTarget. Instead call <c>ConfigureTarget</c> and <c>ConfigureClear</c>.
        // The render pipeline will ensure target setup and clearing happens in an performance manner.
        protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
        {
            // Create render targets for all cases
            m_RenderTarget = RTHandles.Alloc(Vector2.one,
                colorFormat: GraphicsFormat.R16G16B16A16_SFloat, useDynamicScale: true,
                depthBufferBits: DepthBits.None, msaaSamples: MSAASamples.None,
                filterMode: FilterMode.Point, wrapMode: TextureWrapMode.Clamp, name: "_GaussianSplatRT");
                
            m_LeftEyeRT = RTHandles.Alloc(Vector2.one,
                colorFormat: GraphicsFormat.R16G16B16A16_SFloat, useDynamicScale: true,
                depthBufferBits: DepthBits.None, msaaSamples: MSAASamples.None,
                filterMode: FilterMode.Point, wrapMode: TextureWrapMode.Clamp, name: "_LeftEyeRT");
                
            m_RightEyeRT = RTHandles.Alloc(Vector2.one,
                colorFormat: GraphicsFormat.R16G16B16A16_SFloat, useDynamicScale: true,
                depthBufferBits: DepthBits.None, msaaSamples: MSAASamples.None,
                filterMode: FilterMode.Point, wrapMode: TextureWrapMode.Clamp, name: "_RightEyeRT");
        }

        protected override void Execute(CustomPassContext ctx)
        {
            var cam = ctx.hdCamera.camera;

            var system = GaussianSplatRenderSystem.instance;
            if (!system.GatherSplatsForCamera(cam))
                return;

            // Check if we're in VR/stereo mode
            bool isStereo = XRSettings.enabled && cam.stereoEnabled;

            if (isStereo)
            {
                // Left eye rendering
                ctx.cmd.SetGlobalTexture(m_LeftEyeRT.name, m_LeftEyeRT.nameID);
                CoreUtils.SetRenderTarget(ctx.cmd, m_LeftEyeRT, ctx.cameraDepthBuffer, ClearFlag.Color, new Color(0, 0, 0, 0));
                Material matComposite = GaussianSplatRenderSystem.instance.SortAndRenderSplats(cam, ctx.cmd, 0);

                // Right eye rendering
                ctx.cmd.SetGlobalTexture(m_RightEyeRT.name, m_RightEyeRT.nameID);
                CoreUtils.SetRenderTarget(ctx.cmd, m_RightEyeRT, ctx.cameraDepthBuffer, ClearFlag.Color, new Color(0, 0, 0, 0));
                GaussianSplatRenderSystem.instance.SortAndRenderSplats(cam, ctx.cmd, 1);

                // Set both textures for the composite shader
                ctx.cmd.SetGlobalTexture("_LeftEyeTex", m_LeftEyeRT);
                ctx.cmd.SetGlobalTexture("_RightEyeTex", m_RightEyeRT);

                // Compose
                ctx.cmd.BeginSample(GaussianSplatRenderSystem.s_ProfCompose);
                CoreUtils.SetRenderTarget(ctx.cmd, ctx.cameraColorBuffer, ClearFlag.None);
                CoreUtils.DrawFullScreen(ctx.cmd, matComposite, ctx.propertyBlock, shaderPassId: 0);
                ctx.cmd.EndSample(GaussianSplatRenderSystem.s_ProfCompose);
            }
            else
            {
                // Legacy single-eye rendering for backward compatibility
                ctx.cmd.SetGlobalTexture(m_RenderTarget.name, m_RenderTarget.nameID);
                CoreUtils.SetRenderTarget(ctx.cmd, m_RenderTarget, ctx.cameraDepthBuffer, ClearFlag.Color, new Color(0, 0, 0, 0));

                // Add sorting, view calc and drawing commands for each splat object
                Material matComposite = GaussianSplatRenderSystem.instance.SortAndRenderSplats(cam, ctx.cmd, 0);

                // Compose
                ctx.cmd.BeginSample(GaussianSplatRenderSystem.s_ProfCompose);
                CoreUtils.SetRenderTarget(ctx.cmd, ctx.cameraColorBuffer, ClearFlag.None);
                CoreUtils.DrawFullScreen(ctx.cmd, matComposite, ctx.propertyBlock, shaderPassId: 0);
                ctx.cmd.EndSample(GaussianSplatRenderSystem.s_ProfCompose);
            }
        }

        protected override void Cleanup()
        {
            m_RenderTarget.Release();
            m_LeftEyeRT.Release();
            m_RightEyeRT.Release();
        }
    }
}

#endif
