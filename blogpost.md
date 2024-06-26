# InstanceDiffusion, Fully Automated: Leveraging LLMs for Efficiency in Image Generation.

--- 

### A. Stuger, A. Ibrahimi, L. Becker, R. van Emmerik, J. van der Lee

---
This blog post delves into, replicates, and builds upon the discoveries outlined in the CVPR 2024 paper titled ["InstanceDiffusion: Instance-level Control for Image Generation"](https://arxiv.org/abs/2402.03290) by Wang et al. (2024). Introducing InstanceDiffusion, the paper presents a technique that enhances text-to-image diffusion models by providing precise instance-level control, accommodating various language conditions and adaptable instance location specifications. The authors highlight InstanceDiffusion's superiority over specialized state-of-the-art models, demonstrating notable performance enhancements.

This purpose of this blog is to:
1. Help other researchers understand the InstanceDiffusion model. 
2. Verify the authors' claims by reproducing the results from the paper. 
3. Provide an extension to the original model by leveraging LLMs for Modular Efficiency.


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
This block is added between the self-attention and cross-attention layers of the backbone which is crucial in leveraging the model's architecture and therefore enabling image generation that adhere to specific instance attributes and locations , similar to how Flamingo (VLM) [8] proccesses vision and language inputs. The main aim of this block is to tokenize the per-instance conditions and fuse them together with the visual tokens obtained through the frozen text-to-image model. One of the key operations in this block is the *location parmeterization* which is responsible for converting the 4 location formats into 2D points. The *instance tokenizer* then converts these 2D point coordinates for each location using the Fourier mapping. Moreover, it encodes the text prompt using a CLIP text encoder , concatenates the location & text embeddings and feeds them into a MLP network to obtain a single token embedding for the instances.

##### ScaleU Block
This block contains 2 learnable,channel-wise scaling vectors for the main & skip connected features. These vectors are then incorporated into each of the UNet's decoder blocks which lead to an increase in the number of parameters and performance gains. 

##### Multi-Instance Sampler Block
This block is used as a strategy to minimize information leakage across multi-instance conditioning during model inference. For each n amount of instances, a seperate denoising operation for a number of steps is used to retrieve the instance latents. The denoised instance latents for each of the n 
objects  are then integrated with the global latent by averaging the latents togther.

### Related work
##### 1. Image Diffusion Models
Image diffusion models have the ability to produce high-quality images through reiterated denoising. This has drawn significant interest in recent years. These models, including Denoising Diffusion Probabilistic Models (DDPMs) and score-based generative models (SGMs), work by progressively adding noise to an image and then learning to reverse this process to generate new samples \[1\]. 
###### 1.1 Denoising Diffusion Probabilistic Models (DDPMs)
In more detail the Denoising Diffusion Probabilistic Models (DDPMs) makes use of two Markov chains (a forward chain, and reverse chain). The forward chain’s aim is to transform the data to noise while the reverse chain converts the noise back to data by learning transitional kernels that are parameterized by deep neural networks. Formally, the factorization of the joint distribution (x1, x2…xT) which is  conditioned on x0 can be constructed as the following : 

<!--
$$\begin{align} 
q\left(\mathbf{x}_1, \ldots, \mathbf{x}_T \mid \mathbf{x}_0\right) 
= \prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)
\end{align}$$
-->

![Equation](https://cdn.mathpix.com/snip/images/LK_trwHEV9QI_ixAFSbbh8GkV0Q2_3YWR_GtERIeQyo.original.fullsize.png)

A distinguishable property of diffusion models can be found in the approximate posterior ![Inline Equation](https://latex.codecogs.com/png.latex?q%5Cleft%28%5Cmathbf%7Bx%7D_t%20%5Cmid%20%5Cmathbf%7Bx%7D_%7Bt-1%7D%5Cright%29), called the forward process or diffusion process, where it is  fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule: 

<!--
$$ \left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I} \right) \qquad \qquad \text{(Equation 2)}$$
-->
![Equation](https://cdn.mathpix.com/snip/images/lNevS6zD4iXjiSkN0xn8O44xPM6VGUAHTNhQv5jvp90.original.fullsize.png)


The reverse chain process , on the other hand, can be modeled as the following:
<!--
$$\begin{align}
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \mu_\theta\left(\mathbf{x}_t, t\right), \Sigma_\theta\left(\mathbf{x}_t, t\right)\right) \qquad \qquad \text{(Equation 3)}
\end{align}$$
-->
![Equation](https://cdn.mathpix.com/snip/images/L_ErlS8IkchfcbKYXP3cxQBv1fAj7OH1BdTzwAqndMc.original.fullsize.png)

Where  the mean $\mu_\theta$ and variance $\Sigma_\theta$ are parametrized by deep neural networks and  the data sample $x_0$ is generated by sampling a noise vector $x_T$. This process iteratively samples from the learnable transition kernel $(\mathbf{x}_{t-1}$ until t=1 or in other words until it is able to retrieve the original data distribution. The training of these DDPMs on the other hand require the optimization of a variational lower bound on the negative log-likelihood of the data : 

<!--
$$
\mathbb{E}_{t \sim \mathcal{U} \| 1, T \rrbracket, \mathbf{x}_0 \sim q\left(\mathbf{x}_0\right), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})}\left[\lambda(t)\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right] \qquad \qquad \text{(Equation 4)}
$$
-->
![Equation](https://cdn.mathpix.com/snip/images/w814M82B2dpTEDSBuuK9mOjwlu4J2uXiHP8gW99JIi0.original.fullsize.png)

As highlighted in the paper [1], the $\lambda(t)$ serves  as a  positive weighting function , $x_t$ is computed from $x_0$ and $\epsilon$ , and $U[\[ 1, T ]\]$ is a uniform distribution over the set {1,2,…,T}. This objective in the end  and has similar  form as the loss of denoising score matching over multiple noise scales for training score-based generative models. 

###### 1.2 Score-Based Generative Models (SGMs)

A distinctive feature of SGMs ,highlighted  in [1], is the score function (a.k.a stein score) it utilizes. According to the authors, this function is defined as the gradient of the log probability density,  where it is considered as a function of the data x rather than the model parameter $\theta$. These models progressively add stronger gaussian noise to the data while training a deep neural network in order to estimate the score functions of the noisy data distributions. The sample generation process is enabled by applying score function in reverse order by a variety of techniques (e.g. Langevin Monte Carlo). Notably, the training and sampling are decoupled/indepedning processes thereby allowing a flexibility in the sampling methods after the estimation of score functions. 

Recent developments have aimed at enhancing the efficiency and quality of these models by tackling challenges such as computational complexity and inference speed, which are crucial for practical uses like deployment on mobile devices. Diffusion models have also been applied to a range of image editing tasks, showcasing impressive capabilities in creating and altering visual content. For instance, techniques such as StrDiffusion use structure-guided denoising to improve semantic consistency in image inpainting \[2\]. Moreover, new methods are being developed to reduce memorization in text-to-image models, ensuring that the generated images do not too closely resemble the training data. This enhances both originality and privacy \[3\].

##### 2.Text-to-Image Diffusion Models

A proposed pipeline that enhances spatial comprehension and attribute assignment was extensively covered in [5]. Their pipeline (CompFuser) overcomes the limitation of existing text-to-image diffusion models by *iteratively* decoding the generation of multiple objects. The iterative steps firstly involve  generating a single object and then editing the image by placing additional objects in their designated positions. A previous work which the CompFuser was inspired from was the InstructPix2Pix [7], where it synthesizes a large dataset of image editing examples and uses it to train a text-to-image diffusion model. However, it fails at adhering to instructions that involve spatial reasoning. In more detail, the two enhancements that the CompFuser model focuses on is the attribute assignment where the accurate linking of attributes with their respective entities is enabled and the spatial comprehension feature which contains terms that aid in describing objects relative positioning. 

As mentioned above, an iterative generation process was used. This was one of the main contributions of this pipeline, alongside the insertion of additional objects into existing scenes via LLM-grounded Diffusion [6]. During their dataset synthesis phase, the authors introduce a 3-stage generation process. The first stage, caption generation, aims at imitating spatial reasoning scenarios via captions. These captions contain 2 objects (placed left/right) and are preceded by an adjective. The generation of caption was enabled through the deployment of LLMs which were controlled through a detailed task description prompt. The second stage, layout generation, involved instance-level annotations that specified the location and contents  of bounding boxes where they ensured that every object instance was associated to a single bounding box. Lastly, the image synthesis stage, involved a *novel* energy function which preserved the background details of the input image. More specifically, the preservation of the background detail of the first and second image was enabled through the transfer of the reference cross attention map $A^{(1)}$ to the target cross attention map  $A^{(2)}$. These maps are defined as : 
<!--
$$
\mathbf{A}_{u v}^{(i)}=\operatorname{Softmax}\left(\mathbf{q}_u^{\top} \mathbf{k}_v\right) \qquad \qquad \text{(Equation 5)}
$$
-->
![Equation](https://cdn.mathpix.com/snip/images/8Yw2OHGxkJL85ZpqQFSh5SnzoY-wrtjyTsGRMjZuRBU.original.fullsize.png)

Where *u* denotes the spatial location while k_v represents the key corresponding to the feature at token index v , these maps are then used by the model to inspect how these locations relate to the specific features in the text prompt which allows the careful alignment of the data with the elements in the text. These maps where then used in the author’s defined energy function : 
<!--
$$
E\left(\mathbf{A}^{(1)}, \mathbf{A}^{(2)}\right)=\frac{1}{2}\left\|\left(1-m_2\right) \cdot\left[\sum_{u, v \in V} \mathbf{A}_{u v}^{(1)}-\sum_{u, v \in V} \mathbf{A}_{u v}^{(2)}\right]\right\|^2 \qquad \qquad \text{(Equation 6)}
$$
-->
![Equation](https://cdn.mathpix.com/snip/images/Oq-eEfZ15U_-wzHU1l3WFliYqzBGbEqWT8bTRZsj8NQ.original.fullsize.png)

The function above aims at aligning the target cross attention map $A^{(2)}$ with the reference one $A^{(1)}$ for identical tokens (V) and pixels that are associated with both images. 
While this pipeline and existing work, effectively managed to outperform text-to-image and text-and-image-to-image models, it is currently limited to only generating up to 2 objects and therefore struggles with handling occlusions. Moreover, the authors suggested utilizing a multimodal modal such as GPT-4 to improve the image-layout generation process.

##### 3. LLM-grounded Diffusion

As mentioned above, the data generation pipeline of the CompFuser method has been largely influenced and relied on the LLM-grounded diffusion model [7]. This approach essentially augments text-to-image generation with LLM. It uses the LLM to construct an image layout from text-description. This text description serves essentially as an  instruction-based scene specification which enables  broader language support in the prompts. The LMD method is composed of 2 main stages (text-grounded layout generation, layout-grounded image generation). The first stage , text-grounded layout generation, involves the layout representation, text-instruction, and in-context learning phase. As was the case in CompFuser [5] pipeline ,layout-representation  consists of background and bounding boxes for the objects.The text-instructions on the other hand consist of two parts (task specification, supporting details). Lastly, the in-context learning phase was used to provide the LLM with examples after the task description for it to ensure precise layout control.

To generate images ,on the other hand, LMD method applies a layout-grounded stable diffusion. Previous methods that applied semantic guidance,highlighted in paper, were seen to lack the ability of controlling the amount of objects within a semantic region. This issue was seen to occur with instances that were indistinguishable from each-other. Therefore by initially generating masked latents per individual bounding box and then composing the masked latents as priors to guide the image generation overall, the LMD method enabled the precise placement and attribute binding per object instance. 


## <a name="reproduction">Reproduction of the Experiments</a>
To reproduce the experiments described in the paper, we followed a process designed to ensure comparability with the original study. Our approach involved utilizing the same datasets, model configurations, and evaluation metrics as outlined by the authors. To verify our results, we conducted extensive evaluations using the same metrics and datasets as the original study, measuring alignment to instance locations and constancy to specified attributes. Please refer to our [README.md](README.md) for an extensive explanation on how to reproduce our results.

For the evaluation, we measure how well the objects in the generated image adhere to different location formats in the input. 


### Dataset and Evaluation Metrics

For our evaluations we utilized the COCO dataset, a large-scale object detection, segmentation, and captioning dataset. Specifically, we used 500 images from the evaluation set, which constitutes only 10% of the original dataset. This subset was chosen due to the computational expense of generating new images using InstanceDiffusion. The COCO dataset is known for its diversity and complexity, making it an excellent benchmark for evaluating object detection and segmentation models. Our evaluation metrics include Average Precision (AP) and Average Recall (AR) for bounding boxes and instance masks, using the COCO's official metrics, along with the newly introduced Points in Mask (PiM) for evaluating scribbles and single points. Additionally, we assessed attribute binding accuracy using a combination of YOLOv8-Det for detecting bounding boxes and CLIP for predicting and measuring attribute adherence.


### Bounding Box & Instance Mask
For the inputs of 'Bounding boxes' and 'Instance Masks', we evaluate the results using a pretrained YOLOv8m-Det \[4\] detection model. We compare the bounding boxes on the generated image that are detected by the model with the bounding boxes specified in the input using COCO’s official evaluation metrics of AP and AR. In the InstanceDiffusion paper, the authors report the Fréchet Inception Distance (FID) values of various methods, but we were unable to reproduce these values due to insufficient methodological details and lack of provided code. FID is an important metric in generative modeling that measures the similarity between real and generated images by comparing the mean and covariance of their feature vectors. The inability to reproduce the reported FID values raises concerns about the reliability of the results and hinders the validation and comparison of different methods within the research community. However, since FID is not the most important value for reproducing the paper results, we are still able to assess the overall effectiveness and contributions of the InstanceDiffusion approach by focusing on other qualitative and quantitative evaluations provided in the paper.

As seen from Table 1 and Table 2, our reproduction of the results from the paper resulted in significantly better values than the results from the paper. We believe this is caused by the fact that we only utilize 10% of the images used in the original experiments, which might not be a fair representation of the evaluation. Due to time constraints and limited resources, we were unable to run our experiments on the entire dataset while the generation of new images using InstanceDiffusion is computationally expensive. 


<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>box</sub></th>
      <th>AP<sub>box</sub><sup>50</sup></th>
      <th>AR<sub>box</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>38.8</td>
    <td>55.4</td>
    <td>52.9</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>49.9</td>
    <td>66.8</td>
    <td>68.6</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+11.1</td>
    <td>+11.4</td>
    <td>+15.7</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 1.</b> Evaluating different location formats when generating images of reproduction experiments using Bounding Boxes as input.</td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>mask</sub></th>
      <th>AP<sub>mask</sub><sup>50</sup></th>
      <th>AR<sub>mask</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>27.1</td>
    <td>50.0</td>
    <td>38.1</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>40.8</td>
    <td>63.5</td>
    <td>56.0</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+13.7</td>
    <td>+13.5</td>
    <td>+17.9</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 2.</b> Evaluating different location formats when generating images of reproduction experiments using Instance Masks as input.</td>
  </tr>
</table>

In the image below, we present two random samples generated by using either bounding boxes or instance masks as input and compare them with the original image used as ground truth. 

<p align="center">
  <img src="src/data/images/demo_examples/bbox-mask/bbox-mask_examples.png" width=95%>
</p>

### Scribble & Single-point
We measure the alignment performance of using scribbles as input by reporting "Points in Mask" (PiM), a new evaluation metric introduced by the authors of InstanceDiffusion that uses YOLOv8m-Seg and measures how many of randomly sampled points in the input scribble lie within the detected mask. Single-point evaulation is similar to scribble: the instance-level accuracy PiM is 1 if the input point is within the detected mask, and 0 otherwise. We calculate the average PiM score. As seen from the table, we reproduce significantly lower results than the original results from the paper. This be explained by the fact that we only utilize 10% of the images used in the original experiments, which might not be a fair representation of the evaluation, and by the fact that the original 'eval_PiM.py' script of InstanceDiffusion contained multiple errors when calculating the average PiM scores. We were able to debug these errors, but this might have lead to a different implementation than originally meant by the authors. For this reason we encourage the authors of the InstanceDiffusion paper to have a second look at their code, such that future reproduction works will be able to more effectively reproduce their results. 

<table align="center">
  <tr align="center">
      <th align="left" rowspan="2">Method</th>
      <th colspan="1">Points</th>
      <th colspan="1">Scribble</th>
  </tr>
  <tr align="center">
      <th>PiM</th>
      <th>PiM</th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>81.1</td>
    <td>72.4</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>33.66</td>
    <td>23.56</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>-47.44</td>
    <td>-48.84</td>
  </tr>
      <tr align="left">
        <td colspan=7><b>Table 3.</b> Evaluating different location formats as input when generating images of reproduction experiments for points and scribbles.</td>
      </tr>
</table>

In the image below, we present two random samples generated by using either points or scribbles as input and compare them with the original image used as ground truth.

<p align="center">
  <img src="src/data/images/demo_examples/point-scribble/point-scribble_examples.png" width=95%>
</p>



### Compositional attribute binding

Using YOLOv8-Det to detect bounding boxes, we measure how well the generated images adhere to the attribute specified in the instance prompt (color or texture). The cropped bounding box is fed to the CLIP model that predicts its attribute and measures the accuracy of the prediction with respect to the attribute specified in the instance prompt. For the attributes, the 8 common colors of *black, white, red, green, yellow, blue, pink & purple* and the 8 common textures of *rubber, fluffy, metallic, wooden, plastic, fabric, leather & glass* are used. We report the local-CLIP score that measures the distance between the instance text prompt’s features and these cropped object images. See table 4 for the measured Accuracy and local CLIP scores. As seen from the table, we were able to closely replicate the results from the paper, providing a solid justification for the original results of the compositional attribute binding.

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

In the image below, we present two random samples generated by specifying either the texture or color attribute in the instance prompt and compare them with the original image. As seen in the image, specifying these attributes is the cause for interesting outputs. 

<p align="center">
  <img src="src/data/images/demo_examples/color-texture/color-texture_examples.png" width=95%>
</p>


## Leveraging LLM's for Modular Efficiency in InstanceDiffusion. 
The InstanceDiffusion Model uses global descriptions combined with instance descriptions and instance bounding boxes as input for image generation. We adopt this approach but automate the input generation using a Large Language Model (LLM). This automation streamlines the process and creates novel image descriptions, eliminating the need for manually generated input data or reliance on original images from which bounding boxes and descriptions are typically extracted. Notably, the authors of the original paper utilized this latter method using Grounded-SAM. Specifically, we implement a ChatGPT-4o LLM submodule that generates image descriptions and bounding boxes of scenes similar to those of the original COCO dataset. The generated data is then fed into the Instance Diffusion Model to produce images, enhancing the overall process's efficiency.

Our approach aligns with the techniques presented by Derakhshani et al \[5\]. They propose CompFuser to enhance spatial comprehension and attribute assignment in text-to-image generative models by interpreting instructions that define spatial relationships between objects and generate images accordingly. This is achieved through an iterative process that first generates a single object and then edits the image to place additional objects in their designated positions, improving spatial comprehension and attribute assignment.

Similarly, our methodology leverages a Large Language Model (LLM) to automate the generation of image descriptions and bounding boxes. By integrating these into the Instance Diffusion Model, we control spatial arrangement and attributes of individual instances in the generated images. Like CompFuser, our approach addresses the challenge of understanding and executing complex prompts involving multiple objects and their spatial relationships. 
<br>
<p align="center">
  <img src="https://github.com/Jellemvdl/InstanceDiffusion-extension/assets/71041391/a75de5fe-ebbf-439b-a0be-488aed425710" width="600">
</p>

In summary, implementing the dedicated language generation submodule, enhances efficiency and allows for scalability and adaptability within the model architecture, whilst producing novel or possible even customized images. This architecture is a novel, flexible solution for image generation tasks, promising advancements in both research and practical applications.



## Evaluation of LLM Submodule

To evaluate the LLM submodule's input data generation, we used three approaches: firstly is a multiple raters assessed the realism of the generated images, secondly, we scored the global description-to-image alignment using CLIP. The third approach involved deploying a robust Vision Language Model (VLM) called CogVLM, which determines whether the generated images match the input descriptions and bounding boxes.

**Human evaluation: Image realism.** For the human evaluation, we generated 100 images that were scored for realism between 1 and 5, with 5 being the highest possible score. The raters' evaluation focused on three main criteria: (1) how well instance types fit the global scene description (e.g., a coffee table fitting a living room scene), (2) the quality of generations, and (3) the object sizes and their arrangement into a realistic perspective. The raters perceived the image quality similarly, with an average score of 2.38 and individual raters' average scores being closely aligned, namely falling between 2.20 and 2.48. Points were typically lost on criteria 2 and 3, as some generated images displayed instances with unrealistic features, such as distorted physical characteristics of humans or animals or unsmooth transitions between different types of flooring. Additionally, many images showed objects that were not arranged logically in space, with object sizes not matching perspective or objects being cut off.  However, the instances consistently matched each other and the global scene well.
<br><br>
Distribution of scores assigned to the 100 images by human raters:

<img src="https://github.com/Jellemvdl/InstanceDiffusion-extension/blob/main/src/data/images/LLM/ratings_freq.png" alt="Ratings Frequency" width="500"/>

Moreover, we observed that not all objects described by the LLM input were generated. This was exasterbated when the LLM struggled effectively distributing objects across the scene or when arrangements were illogical. 
<br><br>
The following are two example images demonstrate these problems:

<img src="https://github.com/Jellemvdl/InstanceDiffusion-extension/blob/main/src/data/images/LLM/bbox_vs_generation.png" alt="" width="700"/>

The errors in criterion 3 indicate that the LLM struggles with generating bounding boxes for realistic scenes. While the length and width of the bounding boxes matched the proportions of the instances (e.g., a traffic light would be taller than it is wide), their relative size and arrangement were often flawed. For example, a tractor in the distance might appear larger than a nearby car, violating perspective rules. <br><br>The following figure shows examples of differently scoring images that illustrate these generation issues:
<br>
![](https://github.com/Jellemvdl/InstanceDiffusion-extension/blob/main/src/data/images/poor-medium-high.png?raw=true)

Errors in criterion 2 could be due to the LLM's difficulty in maintaining consistent instance quality across different scenes. The LLM might generate high-quality instances in isolation but fail to integrate them smoothly into a coherent scene, leading to mismatched textures and inconsistent lighting.

The bounding box arrangement issues could stem from the LLM's limited understanding of spatial relationships and perspective in three-dimensional space. Despite explicit instructions to ensure realistic arrangements, the LLM often lacked the necessary spatial reasoning to accurately place objects relative to each other in a way that maintains a realistic perspective. This is especially problematic in complex scenes with multiple objects at varying distances and orientations. A possible explanation could be that LLMs, including state-of-the-art models like ChatGPT4o which we deployed, are inherently trained on predominantly language-based datasets, rather than spatially oriented ones (Lian et al., 2024), simply causing a mismatch between the bounding box generation task, and the language-focused tasks the LLM was designed ot excel at. 

In conclusion, while the LLM submodule enables customizability of the InstanceDiffusion generations, and creates novel datasets, there are areas for improvement in generating realistic and well-arranged scenes. Further refinements in the LLM's spatial reasoning and perspective understanding are necessary to address these challenges and improve overall image quality.

**CLIP evaluation: Global text-to-image alignment.**
Next, we evaluated global text-to-image alignment using CLIP. CLIP-scoring involves computing the cosine similarity between the features of the global image descriptions (excl. instance descriptions) and the corresponding generated image. The results show an reasonable average CLIP score of 0.278 across 100 images with a standard deviation of 0.028.

This result is especially notable compared to the somewhat lower local attribute scores for color (0.250) and texture (0.225) reported in the original paper, and implies that the LLM generated instances align with the global image description without compromising performance.

**COGVLM evaluation.** The third method involves using CogVLM to assess the alignment between the generated images and the bounding boxes created by ChatGPT. Initially, the generated images are input into CogVLM, which then identifies and outlines the instances within the images with bounding boxes, referred to as predicted bounding boxes. These predicted bounding boxes represent the locations where the Instance Diffusion model has placed the instances. In contrast, the bounding boxes generated by ChatGPT serve as the ground truth, indicating the intended locations for instance generation. To evaluate the accuracy of instance placement by the Instance Diffusion model, we compute the Intersection over Union (IoU) between the predicted bounding boxes from CogVLM and the ground truth bounding boxes from ChatGPT. This IoU metric helps determine the model's precision in generating instances at specific locations as dictated by the bounding boxes created by ChatGPT. 

The prompt we used for generating the scene is located in [`src/lib/llm_extension/chatgpt/chatgpt_run.py`](src/lib/llm_extension/chatgpt/chatgpt_run.py). Moreover, the prompt for generating bounding boxes over the instances that Instance Diffusion created is located in [`src/lib/llm_submodule/cogvlm/cogvlm.py`](src/lib/llm_submodule/cogvlm/cogvlm.py).
Subsequently, we calculate the IoU of the bounding boxes with the generated caption from CogVLM and the bounding boxes of the scene generated by ChatGPT to assess how accurately the Instance Diffusion model places instances in the specified locations. The overall average IoU is 0.42. Here are some results:

<table align="center">
  <tr align="center">
      <th><img src="src/data/images/output_0_48.png"></th>
  </tr>
  <tr align="left">
    <td colspan=2>The image displays two side-by-side visualizations of bounding box predictions and ground truths for object detection, with ChatGPT's bounding boxes in green and CogVLM's bounding boxes in red, showing IoU values for a person and a bicycle on the left, and a bird and a goat (label converted to dog for both cases) on the right.</td>
  </tr>
</table>
<table align="center">
  <tr align="center">
      <th><img src="src/data/images/output_30_12.png"></th>
  </tr>
  <tr align="left">
    <td colspan=2>The image displays two side-by-side visualizations of bounding box predictions and ground truths for object detection, with ChatGPT's bounding boxes in green and CogVLM's bounding boxes in red. The right image shows IoU values for a person (0.59) and a truck (0.00), indicating that Instance Diffusion did not place the truck instance on the specified location. </td>
  </tr>
</table>

As shown in the images Instance Diffusion can generate an instance a majority of the time at the specified location. However, one limitation is the creation of the bounding boxes by ChatGPT. ChatGPT creates rather small bounding boxes that are not really aligned with the size of the instance wanted to be created. As shown in the forklift image the bounding box that ChatGPT created for the forklift is rather small. It is not a realistic size of a forklift and the Instance Diffusion model generates a much larger forklift. Therefor the IoU is very low, however the forklift is generated at the specified location. For many instances this is the case, therefor the IoU value is not very high. It is possible that ChatGPT may not effectively conceptualize a scene and accurately determine the scale of instances within it. Additionally, an another limitation arises from the prompt provided to ChatGPT, which specifies numerous instances to be created. However, CogVLM only recognizes a few of these instances. Consequently, our evaluation is restricted to only those instances identified by CogVLM.


### Strengths, Weaknesses, and Potential of InstanceDiffusion

#### Strengths 

1. **Precision Control**: InstanceDiffusion excels in providing precise instance-level control, allowing users to specify exact locations and detailed descriptions for each instance in the generated image. This level of control is a significant advancement over traditional text-conditioned diffusion models, which often lack the ability to fine-tune individual elements within an image.

2. **Flexibility in Input Specifications**: The model supports various forms of instance location inputs, such as single points, bounding boxes, instance masks, and scribbles. This flexibility makes it highly adaptable to different user needs and use cases, ranging from detailed design applications to creative artwork.

3. **Unified Parameterization**: The method's ability to parameterize diverse forms of instance-level conditions into a unified framework simplifies the model architecture and improves performance. By projecting different condition types into a common feature space, InstanceDiffusion ensures consistent and coherent integration of these conditions during the image generation process.

4. **Performance Enhancements**: The inclusion of UniFusion, ScaleU, and Multi-Instance Sampler blocks leads to significant performance improvements. UniFusion ensures effective fusion of instance-level conditions with visual tokens, ScaleU enhances image fidelity by recalibrating key features, and the Multi-Instance Sampler minimizes information leakage in multi-instance scenarios, ensuring clarity and distinction for each instance.

5. **Superior Image Quality**: Empirical results demonstrate that InstanceDiffusion outperforms state-of-the-art models in generating high-quality images that accurately reflect the specified instance-level conditions. This superiority is particularly evident in scenarios involving multiple instances, where the model maintains clarity and coherence.

#### Weaknesses 

1. **Computational Complexity**: The enhanced capabilities of InstanceDiffusion come at the cost of increased computational complexity. The additional blocks (UniFusion, ScaleU, and Multi-Instance Sampler) add to the model's parameter count, potentially leading to longer training times and higher resource requirements.

2. **Scalability**: While the model performs well with multiple instances, its scalability to very large numbers of instances or highly complex scenes might be limited. Managing and processing numerous instance-level conditions simultaneously can introduce challenges in terms of memory and computational efficiency.

3. **Dependency on Pretrained Models**: InstanceDiffusion relies on a pretrained text-to-image UNet model, which is kept frozen during training. This dependency might limit the model's adaptability to different domains or datasets that significantly differ from the ones used to pretrain the base model.

4. **Complexity of Input Specification**: The requirement for precise input specifications (e.g., bounding boxes, scribbles) might pose a challenge for users who lack the technical expertise or tools to provide such detailed inputs. Simplifying the input specification process could enhance user accessibility and adoption.

#### Potential 

1. **Integration with Large Language Models (LLMs)**: Extending InstanceDiffusion with LLMs like ChatGPT can further enhance its capabilities. For instance, an LLM can be used to generate bounding boxes and scribble points based on textual descriptions provided by the user. This integration can automate and simplify the input specification process, making the model more user-friendly and accessible.

2. **Broader Application Range**: The precise control offered by InstanceDiffusion makes it suitable for a wide range of applications, from graphic design and digital art to more technical fields like medical imaging and autonomous vehicle training. By enabling detailed customization of generated images, the model can cater to diverse industry needs.

3. **Interactive Design Tools**: The model's ability to generate high-quality images based on detailed instance-level conditions can be leveraged to develop interactive design tools. These tools can assist designers in creating complex scenes by providing real-time feedback and suggestions based on the specified conditions.

4. **Enhanced Training Techniques**: Future research could focus on optimizing the training process to reduce computational complexity and improve scalability. Techniques such as model pruning, knowledge distillation, or leveraging more efficient architectures could be explored to make InstanceDiffusion more practical for real-world applications.

5. **Expanding Dataset Diversity**: To mitigate the dependency on pretrained models, further work can involve training InstanceDiffusion on more diverse datasets. This would enhance the model's ability to generalize across different domains and improve its adaptability to various types of input conditions.

By addressing its current weaknesses and leveraging its strengths, InstanceDiffusion has the potential to set new standards in the field of instance-level controlled image generation, paving the way for more sophisticated and user-friendly generative models.

## Conclusion
The instance-diffusion method, as presented and thoroughly exlplained in this blogpost, stands as a testament to the adept utilization of advanced methodologies in the realm of image generation. Our replication study, anchored by the COCO dataset, serves as a robust validation of the method's efficacy and potential. However, despite our conscientious efforts to adhere closely to the original study's protocols, we encountered notable disparities in certain results. These discrepancies primarily stem from our deliberate decision to operate within the constraints of a subset of the dataset, a choice motivated by computational constraints and resource availability. This limitation, while acknowledged, underscores the inherent complexity of replicating experimental findings in real-world contexts. 

Our exploration aimed to address a limitation of the original model by adopting a modular framework and leveraging GPT-4o for generating bounding boxes, thereby mitigating the need for an additional image dataset. While this approach proved convenient and valuable, enabling seamless integration with the model, it revealed a constraint of the Language Model (LLM): the absence of spatial awareness. Consequently, this deficiency yielded suboptimal results in certain instances.

To further research the capabilities of the LLM, we conducted a comparative analysis with CogVLM and employed human annotators for evaluation. Although our approach has demonstrated merits revealed through our examination, significant opportunities for improvement in image generation remain. These findings underscore the fertile ground for future research in exploring techniques to enhance the collaborative interaction between LLMs and image generation models such as InstanceDiffusion

## Authors' Contributions
- Anesa: Background & Related Work Research, assisted Lisann in integrating the LLM submodule into existing pipeline, conducted/participated in the manual evaluation of LLM submodule image-generation. Wrote the Introduction and Related work section.
- Amalia: Researched related work and background for implementing the extension, and assisted the group by formalizing the extension to the original paper. Responsible for researching, formalizing, and simplifying the extension. Participated in the manual evaluation of the LLM submodule for image generation. Authored part of the introduction and visualized the extension and pipeline in the blog post.
- Richter: Assisted Jelle with reproduction results in first weeks. CogVLM submodule implementation plus visualization and evaluation. Wrote about the CogVLM evaluation in the blogpost and Readme. Did preliminary work for the background. Incorporated the jobfiles in the repository. Reproduction evaluation Readme.
- Lisann: Environment setup. LLM submodule implementation (excluding CogVLM), LLM demo, and organisation and analysis of LLM  evaluation through ratings. CLIP evaluation. Wrote LLM Readme and blogpost sections (excluding CogVLM). Tested all instructions and demos described in Readme, and increased robustness of scripts and jobs (e.g., by using dynamic paths).
- Jelle: Responsible for reproduction of original paper's results + demos and the corresponding blogpost section, helped with environment setup. Organized the github repository, including jobs and inference demo scripts, adding setup & runnable scripts to Readme. 

## CO2 Impact
Utilizing approximately 70 GPU hours on the Snellius supercomputer to run our experiments has a notable CO2 impact. Diffusion models are computationally intensive, this combined with the structure of the used data, attributed significantly to this environmental footprint. We acknowledge the substantial carbon emissions associated with these high-performance computations and recognize their detrimental effect on the planet. That is why we have taken careful consideration not to retrain the entire model but to inject our extension into the existing model. Our commitment to advancing machine learning research is paralleled by our awareness of the environmental costs.
## Bibliography

[1] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M.-H. Yang, "Diffusion models: A comprehensive survey of methods and applications," ACM Computing Surveys, vol. 56, no. 4, pp. 1-39, 2023.

[2] [Haipeng Liu, Yang Wang, Biao Qian, Meng Wang, Yong Rui. CVPR 2024, Seattle, USA], "[StrDiffusion]," GitHub repository, [https://github.com/htyjers/StrDiffusion]. Accessed on: [03, 2024].

[3] J. Ren, Y. Li, S. Zen, H. Xu, L. Lyu, Y. Xing, and J. Tang, "Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention," arXiv preprint arXiv:2403.11052, 2024.

[4] G. Jocher, A. Chaurasia, and J. Qiu, "YOLO by Ultralytics," Jan. 2023. GitHub repository, [https://github.com/ultralytics/ultralytics].

[5] M. M. Derakhshani, M. Xia, H. Behl, C. G. M. Snoek, and V. Rühle, "Unlocking Spatial Comprehension in Text-to-Image Diffusion Models," arXiv preprint arXiv:2311.17937, 2023.

[6] L. Lian, B. Li, A. Yala, and T. Darrell, "LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models," arXiv preprint arXiv:2305.13655, 2024.

[7] T. Brooks, A. Holynski, and A. A. Efros, "InstructPix2Pix: Learning to Follow Image Editing Instructions," arXiv preprint arXiv:2211.09800, 2023.

[8] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, "Flamingo: a Visual Language Model for Few-Shot Learning," arXiv preprint arXiv:2204.14198, 2022.

--- 
