---
layout: post
title:  "Scalable Machine Learning New"
date:   2021-01-01 00:00:10 -0030
categories: deeplearning
mathjax: true
---



# Content

1. TOC
{:toc}
---


# Using the Right Processors CPUs, GPUs, ASICs, and TPUs

CPUs are not ideal for large scale machine learning (ML), and they can quickly turn into a bottleneck because of the sequential processing nature. An upgrade on CPUs for ML is GPUs (graphics processing units). Unlike CPUs, GPUs contain hundreds of embedded ALUs, which make them a very good choice for any process that can benefit by leveraging parallelized computations. GPUs are much faster than CPUs for computations like vector multiplications. However, both CPUs and GPUs are designed for general purpose usage and suffer from **von Neumann bottleneck** () and higher power consumption.

> :bulb: **von Neumann bottleneck:** The shared bus between the program memory and data memory leads to the von Neumann bottleneck, the limited throughput (data transfer rate) between the central processing unit (CPU) and memory compared to the amount of memory. 

A step beyond CPUs and GPUs is **ASICs (Application Specific Integrated Chips)**, where we trade general flexibility for an increase in performance. There have been a lot of exciting research on for designing ASICs for deep learning, and Google has already come up with three generations of **ASICs called Tensor Processing Units (TPUs)**.

TPUs exploit the fact that neural network computations are operations of matrix multiplication and addition, and have the specialized architecture to perform just that. TPUs consist of **MAC units (multipliers and accumulators)** arranged in a `systolic array` fashion, which enables **matrix multiplications without memory access**, thus consuming less power and reducing costs.

> :bulb: **systolic array:** In parallel computer architectures, a systolic array is a homogeneous network of tightly coupled data processing units (DPUs) called cells or nodes. Each node or DPU independently computes a partial result as a function of the data received from its upstream neighbors, stores the result within itself and passes it downstream.

This way of performing matrix multiplications also reduces the computational complexity from the order of $n^3$ to order of $3n - 2$. for more details check [here](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu).

**Reference:**

- [Machine Learning: How to Build Scalable Machine Learning Models](https://www.codementor.io/blog/scalable-ml-models-6rvtbf8dsd) :fire:
- [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) :rocket: 


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


-----

# Distributed model training

<center>
<img src="https://cdn.filestackcontent.com/jERYMQCTuWmEdsb3Lrfs" width="600">
</center>

A typical, supervised learning experiment consists of feeding the data via the input pipeline, doing a forward pass, computing loss, and then correcting the parameters with an objective to minimize the loss. Performances of various hyperparameters and architectures are evaluated before selecting the best one.

Let's explore how we can apply the `divide and conquer` approach to decompose the computations performed in these steps into granular ones that can be run independently of each other, and aggregated later on to get the desired result. After decomposition, we can leverage horizontal scaling of our systems to improve time, cost, and performance.


----

:santa: happy coding


----
