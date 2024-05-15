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

**Table 1:** ..

--- 
