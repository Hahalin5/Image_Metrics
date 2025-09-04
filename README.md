# Image Metrics
We often want to know the quality of our images. This can either be to detect and quantify distortions present in the data (noise, blur, compression artifacts, etc.) or to compare a result to some ground truth data.

Note that typically when we use the metrics described below, we are looking at the pixel values of an image. Some other ways to compare or analyze images include:
- looking at the image histogram
- filtering the image (cross correlation, gradient, etc.)
- looking at the Fourier transform of the image.

# Image Comparison Metrics
Also called full-reference quality metrics, these compare some n × m image x to a ground truth reference image y.

1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. Peak Signal-to-Noise Ratio (PSNR)
4. Structural Similarity Index Measure (SSIM)

In this project, I have implemented the python scripts that shows how to use these four different metrics. 

# Reference
https://www.geeksforgeeks.org/maths/mean-squared-error/

https://www.geeksforgeeks.org/python/how-to-calculate-mean-absolute-error-in-python/

https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

https://medium.com/@akp83540/structural-similarity-index-ssim-c5862bb2b520





