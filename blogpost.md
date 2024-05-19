# A Deep Dive Into "InstanceDiffusion: Instance-level Control for Image Generation"

--- 

### A. Stuger, A. Ibrahimi, L. Becker, R. van Emmerik, J. van der Lee

---
This blog post delves into, replicates, and builds upon the discoveries outlined in the CVPR 2024 paper titled ["InstanceDiffusion: Instance-level Control for Image Generation"](https://arxiv.org/abs/2402.03290) by Wang et al. (2024). Introducing InstanceDiffusion, the paper presents a technique that enhances text-to-image diffusion models by providing precise instance-level control, accommodating various language conditions and adaptable instance location specifications. The authors highlight InstanceDiffusion's superiority over specialized state-of-the-art models, demonstrating notable performance enhancements.

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

#### Related work
##### Image Diffusion Models
Image diffusion models have the ability to produce high-quality images through reiterated denoising. This has drawn significant interest in recent years. These models, including Denoising Diffusion Probabilistic Models (DDPMs) and score-based generative models (SGMs), work by progressively adding noise to an image and then learning to reverse this process to generate new samples \[1\]. Recent developments have aimed at enhancing the efficiency and quality of these models by tackling challenges such as computational complexity and inference speed, which are crucial for practical uses like deployment on mobile devices.

Diffusion models have also been applied to a range of image editing tasks, showcasing impressive capabilities in creating and altering visual content. For instance, techniques such as StrDiffusion use structure-guided denoising to improve semantic consistency in image inpainting \[2\]. Moreover, new methods are being developed to reduce memorization in text-to-image models, ensuring that the generated images do not too closely resemble the training data. This enhances both originality and privacy \[3\].


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


## Leveraging LLM's for Modular Efficiency in InstanceDiffusion. 
The Instance Diffusion Model uses global descriptions combined with instance descriptions and instance bounding boxes as input for image generation. We adopt this approach but automate the input generation using a Large Language Model (LLM). This automation streamlines the process, eliminating the need for manually generated input data. Specifically, we implement a GPT-4o LLM submodule that generates image descriptions and bounding boxes similar to the original COCO input. The generated data is then fed into the Instance Diffusion Model to produce images, enhancing the overall process's efficiency.

Our approach aligns closely with the techniques presented by Derakhshani et al \[4\]. They propose CompFuser to enhance spatial comprehension and attribute assignment in text-to-image generative models by interpreting instructions that define spatial relationships between objects and generate images accordingly. This is achieved through an iterative process that first generates a single object and then edits the image to place additional objects in their designated positions, improving spatial comprehension and attribute assignment.

Similarly, our methodology leverages a Large Language Model (LLM) to automate the generation of image descriptions and bounding boxes. By integrating these into the Instance Diffusion Model, we enable precise control over the spatial arrangement and attributes of individual instances in the generated images. Like CompFuser, our approach addresses the challenge of understanding and executing complex prompts involving multiple objects and their spatial relationships. 

![image](https://github.com/Jellemvdl/InstanceDiffusion-extension/assets/71041391/7ab2afaa-8746-4e78-9ebb-e8abe6450181)

Our modular approach, facilitated by a dedicated language generation submodule, enhances efficiency and allows for scalability and adaptability within the model architecture. Incorporating an LLM for generating image descriptions enriches the model's capabilities, enabling it to produce nuanced and contextually relevant single points. This methodology represents a robust and comprehensive solution for image generation tasks, promising advancements in both research and practical applications.



## Evaluation of LLM Submodule

To evaluate the LLM submodule, we used two approaches: CogVLM (...) and a multiple raters' assessment of the realism of the generated images. The raters' evaluation focused on three main criteria: (1) how well instance types fit the global scene description (e.g., a coffee table fitting a living room scene), (2) the quality of generations, and (3) the object sizes and their arrangement into a realistic perspective.

For the manual raters' assessment, we generated 100 images that were scored between 1 and 5, with five being the highest. The raters perceived the image quality similarly, with an average score of 2.38 and individual raters' averages ranging between 2.20 and 2.48. Points were typically lost on criteria 2 and 3, as some generated images displayed instances with unrealistic features, such as distorted physical characteristics of humans or animals or unsmooth transitions between different types of flooring. Additionally, many images had objects that were not arranged logically in space, with object sizes not matching perspective or objects being cut off. However, the instances matched each other and the global scene well.

The errors in criterion 3 indicate that the LLM struggles with generating bounding boxes for realistic scenes. While the length and width of the bounding boxes matched the proportions of the instances (e.g., a traffic light would be taller than it is wide), their relative size and arrangement were often flawed. For example, a tractor in the distance might appear larger than a nearby car, violating perspective rules. Figure [3] shows examples of differently scoring images that illustrate these generation issues.

Errors in criterion 2 could be due to the LLM's difficulty in maintaining consistent instance quality across different scenes. The LLM might generate high-quality instances in isolation but fail to integrate them smoothly into a coherent scene, leading to mismatched textures and inconsistent lighting.

The bounding box arrangement issues could stem from the LLM's limited understanding of spatial relationships and perspective in three-dimensional space. Despite explicit instructions to ensure realistic arrangements, the LLM might lack the necessary spatial reasoning to accurately place objects relative to each other in a way that maintains a realistic perspective. This is especially problematic in complex scenes with multiple objects at varying distances and orientations.

In conclusion, while the LLM submodule significantly enhances the efficiency and scalability of the Instance Diffusion Model, there are areas for improvement in generating realistic and well-arranged scenes. Further refinements in spatial reasoning and perspective understanding are necessary to address these challenges and improve overall image quality.

XXX image of ratings XXX

[CogVLM section]



## Bibliography

[1] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M.-H. Yang, "Diffusion models: A comprehensive survey of methods and applications," ACM Computing Surveys, vol. 56, no. 4, pp. 1-39, 2023.

[2] [Haipeng Liu, Yang Wang, Biao Qian, Meng Wang, Yong Rui. CVPR 2024, Seattle, USA], "[StrDiffusion]," GitHub repository, [https://github.com/htyjers/StrDiffusion]. Accessed on: [03, 2024].

[3] J. Ren, Y. Li, S. Zen, H. Xu, L. Lyu, Y. Xing, and J. Tang, "Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention," arXiv preprint arXiv:2403.11052, 2024.

[4] M. M. Derakhshani, M. Xia, H. Behl, C. G. M. Snoek, and V. Rühle, "Unlocking Spatial Comprehension in Text-to-Image Diffusion Models," arXiv preprint arXiv:2311.17937, 2023.




--- 
