// SPDX-License-Identifier: MIT
#if GS_ENABLE_URP

#if !UNITY_6000_0_OR_NEWER
#error Unity Gaussian Splatting URP support only works in Unity 6 or later
#endif

using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    // Note: I have no idea what is the purpose of ScriptableRendererFeature vs ScriptableRenderPass, which one of those
    // is supposed to do resource management vs logic, etc. etc. Code below "seems to work" but I'm just fumbling along,
    // without understanding any of it.
    //
    // ReSharper disable once InconsistentNaming
    class GaussianSplatURPFeature : ScriptableRendererFeature
    {
        class GSRenderPass : ScriptableRenderPass
        {
            const string GaussianSplatRTName = "_GaussianSplatRT";
            const string LeftEyeRTName = "_LeftEyeTex";
            const string RightEyeRTName = "_RightEyeTex";

            const string ProfilerTag = "GaussianSplatRenderGraph";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);
            static readonly int s_leftEyeRT = Shader.PropertyToID(LeftEyeRTName);
            static readonly int s_rightEyeRT = Shader.PropertyToID(RightEyeRTName);

            class PassData
            {
                internal UniversalCameraData CameraData;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT; // For backward compatibility
                internal TextureHandle LeftEyeRT;
                internal TextureHandle RightEyeRT;
                internal bool IsStereo;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                RenderTextureDescriptor rtDesc = cameraData.cameraTargetDescriptor;
                rtDesc.depthBufferBits = 0;
                rtDesc.msaaSamples = 1;
                rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;

                // Check if we're in VR/stereo mode
                bool isStereo = XRSettings.enabled && cameraData.camera.stereoEnabled;
                Debug.Log($"isStereo: {isStereo}");
                
                // Create render textures
                var gaussianSplatRT = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, GaussianSplatRTName, true);
                var leftEyeRT = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, LeftEyeRTName, true);
                var rightEyeRT = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, RightEyeRTName, true);

                passData.CameraData = cameraData;
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = gaussianSplatRT;
                passData.LeftEyeRT = leftEyeRT;
                passData.RightEyeRT = rightEyeRT;
                passData.IsStereo = isStereo;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture);
                builder.UseTexture(gaussianSplatRT, AccessFlags.Write);
                builder.UseTexture(leftEyeRT, AccessFlags.Write);
                builder.UseTexture(rightEyeRT, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                    
                    if (data.IsStereo)
                    {
                        // Left eye rendering
                        commandBuffer.SetGlobalTexture(s_leftEyeRT, data.LeftEyeRT);
                        CoreUtils.SetRenderTarget(commandBuffer, data.LeftEyeRT, data.SourceDepth, ClearFlag.Color, Color.clear);
                        Material matComposite = GaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer, 0);

                        // Right eye rendering
                        commandBuffer.SetGlobalTexture(s_rightEyeRT, data.RightEyeRT);
                        // CoreUtils.SetRenderTarget(commandBuffer, data.RightEyeRT, data.SourceDepth, ClearFlag.Color, Color.clear);
                        // GaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer, 1);

                        // Set both textures for the composite shader
                        commandBuffer.SetGlobalTexture("_LeftEyeTex", data.LeftEyeRT);
                        commandBuffer.SetGlobalTexture("_RightEyeTex", data.RightEyeRT);
                        
                        // Composite to the final target
                        commandBuffer.BeginSample(GaussianSplatRenderSystem.s_ProfCompose);
                        Blitter.BlitCameraTexture(commandBuffer, data.SourceTexture, data.SourceTexture, matComposite, 0);
                        commandBuffer.EndSample(GaussianSplatRenderSystem.s_ProfCompose);
                    }
                    else
                    {
                        // Legacy single-eye rendering for backward compatibility
                        commandBuffer.SetGlobalTexture(s_gaussianSplatRT, data.GaussianSplatRT);
                        CoreUtils.SetRenderTarget(commandBuffer, data.GaussianSplatRT, data.SourceDepth, ClearFlag.Color, Color.clear);
                        Material matComposite = GaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer, 0);
                        
                        commandBuffer.BeginSample(GaussianSplatRenderSystem.s_ProfCompose);
                        Blitter.BlitCameraTexture(commandBuffer, data.GaussianSplatRT, data.SourceTexture, matComposite, 0);
                        commandBuffer.EndSample(GaussianSplatRenderSystem.s_ProfCompose);
                    }
                });
            }
        }

        GSRenderPass m_Pass;
        bool m_HasCamera;

        public override void Create()
        {
            m_Pass = new GSRenderPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            };
        }

        public override void OnCameraPreCull(ScriptableRenderer renderer, in CameraData cameraData)
        {
            m_HasCamera = false;
            var system = GaussianSplatRenderSystem.instance;
            if (!system.GatherSplatsForCamera(cameraData.camera))
                return;

            m_HasCamera = true;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (!m_HasCamera)
                return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass = null;
        }
    }
}

#endif // #if GS_ENABLE_URP
