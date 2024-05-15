# A Deep Dive Into "InstanceDiffusion: Instance-level Control for Image Generation"

--- 

### A. Stuger, A. Ibrahimi, L. Becker, R. van Emmerik, J. van der Lee

---
This blog post delves into, replicates, and builds upon the discoveries outlined in the CVPR 2024 paper titled "InstanceDiffusion: Instance-level Control for Image Generation."["InstanceDiffusion: Instance-level Control for Image Generation"](https://arxiv.org/abs/2402.03290) by Wang et al. (2024). Introducing InstanceDiffusion, the paper presents a technique that enhances text-to-image diffusion models by providing precise instance-level control, accommodating various language conditions and adaptable instance location specifications. The authors highlight InstanceDiffusion's superiority over specialized state-of-the-art models, demonstrating notable performance enhancements.

This purpose of this blog is to:
1. Help other researchers understand the InstanceDiffusion model. 
2. Verify the authors' claims by reproducing the results from the paper. 
3. Provide an extension to the original model.



## <a name="reproduction">Reproduction of the Experiments</a>

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
</table>

**Table 1:**  Evaluating different location formats as input when generating images of reproduction experiments Boxes.

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
</table>

**Table 2:** Evaluating different location formats as input when generating images of reproduction experiments Instance Maks.

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
</table>

**Table 3:** Evaluating different location formats as input when generating images of reproduction experiments for points and scribbles.

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
</table>

**Table 4:** Attribute binding reproduction results for color and texture.

--- 
