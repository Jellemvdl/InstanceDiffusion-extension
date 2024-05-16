# A Deep Dive Into "InstanceDiffusion: Instance-level Control for Image Generation"

--- 

### A. Stuger, A. Ibrahimi, L. Becker, R. van Emmerik, J. van der Lee

---
This blog post delves into, replicates, and builds upon the discoveries outlined in the CVPR 2024 paper titled "InstanceDiffusion: Instance-level Control for Image Generation."["InstanceDiffusion: Instance-level Control for Image Generation"](https://arxiv.org/abs/2402.03290) by Wang et al. (2024). Introducing InstanceDiffusion, the paper presents a technique that enhances text-to-image diffusion models by providing precise instance-level control, accommodating various language conditions and adaptable instance location specifications. The authors highlight InstanceDiffusion's superiority over specialized state-of-the-art models, demonstrating notable performance enhancements.

This purpose of this blog is to:
1. Help other researchers understand the InstanceDiffusion model. 
2. Verify the authors' claims by reproducing the results from the paper. 
3. Provide an extension to the original model.


## <a name="introduction">Introduction</a>
Existing and previous work which focus on text-conditioned diffusion models enable a high quality image production, however, according to the authors this text-based control does not always allow precise or intuitive control over the output image. The reason why control is such a favorable and desired attribute is due to the broad range of design applications for generative models. Therefore, authors proposed and studied instance-conditioned image generation where the user can specify **every** instance in terms of its *location* (via single point, bounding box, instance mask,scribble) and _instance-level text prompt_. This in return, allows for flexible input through various instance location specification and more fine-grained contol over the instance's attributes. Moreover, the author's method presents a unified way to parameterize their information during the generation process which is a simpler & better performing method. In more detail, the superiority of the InstanceDiffusion method over existing models can be accredited to the **3** major changes proposed by the authors which altered the text-to-image models and enabled the precise instance-level control:
1. **UniFusion**: enables instance-level conditions for text-to-image models

    a. projects various forms of instance-level conditions into the same feature space

    b. injects the instance-level layout and descriptions into the visual tokens

2. **ScaleU**: improves image fidelity

    a. recalibrates the main features/ low frequency components within the skip connection features of UNet

3. **Multi-instance Sampler** : improves generations for multiple instances


   a. reduces information leakage and confusion between the conditions on multiple instances


#### Model Architecture & Details
To enable a more generic and flexible scene-layout control in terms of location and attributes of the instances, the authors focused on 2 conditioning inputs for each instance (1.Location + 2. Text Caption describing the instance). The authors then used a pretrained text-to-image UNet model that is kept frozen and added the 3 above-mentioned learnable blocks (UniFusion, ScaleU,Multi-Instance
Sampler). 
##### UniFusion Block
This block is added between the self-attention and cross-attention layers of the backbone. The main aim of this block is to tokenize the per-instance conditions and fuse them together with the visual tokens obtained through the frozen text-to-image model. One of the key operations in this block is the *location parmeterization* which is responsible for converting the 4 location formats into 2D points. The *instance tokenizer* then converts these 2D point coordinates for each location using the Fourier mapping. Moreover, it encodes the text prompt using a CLIP text encoder , concatenates the location & text embeddings and feeds them into a MLP network to obtain a single token embedding for the instances.

##### ScaleU Block
This block contains 2 learnable,channel-wise scaling vectors for the main & skip connected features. These vectors are then incorporated into each of the UNet's decoder blocks which lead to an increase in the number of parameters and performance gains. 

##### Multi-Instance Sampler Block
This block is used as a strategy to minimize information leakage across multi-instance conditioning during model inference. For each n amount of instances, a seperate denoising operation for a number of steps is used to retrieve the instance latents. The denoised instance latents for each of the n 
objects  are then integrated with the global latent by averaging the latents togther.

## Leveraging LLM's for a modular efficiency in InstanceDiffusion. 
The Instance Diffusion Model utilizes bounding boxes for image generation. This approach is adopted to circumvent the necessity of retraining the entire model. Instead, a submodule incorporating a LLM is devised to facilitate bounding box generation. Subsequently, these bounding boxes are inputted into the Instance Diffusion Model, thereby enhancing the efficiency of the overall process.

Similarly, a comparable technique is employed in the treatment of single points within the model. Herein, a Large Language Model (LLM) once again furnishes the requisite captions for the generation of single points. These captions are then seamlessly integrated into the model, thereby ensuring a comprehensive and cohesive treatment of both bounding boxes and single points. 

![image](https://github.com/Jellemvdl/InstanceDiffusion-extension/assets/71041391/7ab2afaa-8746-4e78-9ebb-e8abe6450181)

Our modular approach, facilitated by a dedicated language generator submodule, not only enhances efficiency but also opens avenues for scalability and adaptability within the model architecture.The incorporation of a Language and Linear Model (LLM) for generating captions, further enriches the model's capabilities, enabling it to produce nuanced and contextually relevant single points. Overall, this amalgamation of methodologies represents a robust and comprehensive solution for image generation tasks, promising advancements in both research and practical applications.


## <a name="reproduction">Reproduction of the Experiments</a>
To reproduce the experiments described in the paper, we followed a process designed to ensure comparability with the original study. Our approach involved utilizing the same datasets, model configurations, and evaluation metrics as outlined by the authors. To verify our results, we conducted extensive evaluations using the same metrics and datasets as the original study, measuring alignment to instance locations and constancy to specified attributes. However, we were unable to reproduce the FID values reported in the paper because the author did not mention this metric on the project's GitHub page, nor was there any code provided for its computation.

<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>box</sub></th>
      <th>AP<sub>box</sub><sup>50</sup></th>
      <th>AR<sub>box</sub></th>
      <th>FID (↓)</th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>38.8</td>
    <td>55.4</td>
    <td>52.9</td>
    <td>23.9</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>49.9</td>
    <td>66.8</td>
    <td>68.6</td>
    <td>..</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+11.1</td>
    <td>+11.4</td>
    <td>+15.7</td>
    <td>..</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 1.</b> Evaluating different location formats as input when generating images of reproduction experiments Boxes.</td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>mask</sub></th>
      <th>AP<sub>mask</sub><sup>50</sup></th>
      <th>AR<sub>mask</sub></th>
      <th>FID (↓)</th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>27.1</td>
    <td>50.0</td>
    <td>38.1</td>
    <td>25.5</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>40.8</td>
    <td>63.5</td>
    <td>56.0</td>
    <td>..</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+13.7</td>
    <td>+13.5</td>
    <td>+17.9</td>
    <td>..</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 2.</b> Evaluating different location formats as input when generating images of reproduction experiments Instance Maks.</td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <th align="left" rowspan="2">Method</th>
      <th colspan="2">Points</th>
      <th colspan="2">Scribble</th>
  </tr>
  <tr align="center">
      <th>PiM</th>
      <th>FID (↓)</th>
      <th>PiM</th>
      <th>FID (↓)</th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>81.1</td>
    <td>27.5</td>
    <td>72.4</td>
    <td>27.3</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>..</td>
    <td>..</td>
    <td>..</td>
    <td>..</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+..</td>
    <td>+..</td>
    <td>+..</td>
    <td>+..</td>
  </tr>
      <tr align="left">
        <td colspan=7><b>Table 3.</b> Evaluating different location formats as input when generating images of reproduction experiments for points and scribbles.</td>
      </tr>
</table>

<table align="center">
  <tr align="center">
      <th align="left" rowspan="2">Methods</th>
      <th colspan="2">Color</th>
      <th colspan="2">Texture</th>
  </tr>
  <tr align="center">
      <th>A<sub>cc</sub><sup>color</sup></th>
      <th>CLIP<sub>local</sub></th>
      <th>A<sub>cc</sub><sup>texture</sup></th>
      <th>CLIP<sub>local</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>54.4</td>
    <td>0.250</td>
    <td>26.8</td>
    <td>0.225</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>53.3</td>
    <td>0.248</td>
    <td>26.9</td>
    <td>0.226</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td style="color:red;">-1.1</td>
    <td style="color:red;">-0.002</td>
    <td style="color:green;">+0.1</td>
    <td style="color:green;">+0.001</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 4.</b> Attribute binding reproduction results for color and texture.</td>
  </tr>  
</table>

--- 
