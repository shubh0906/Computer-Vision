Hello Vision World

Write an OpenCV program to do the following things:

    Read an image from a file and display it to the screen

    Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
    Resize the image uniformly by ½

Filters:
The goal in this assignment is to get you acquainted with filtering in the spatial domain as well as in the frequency domain.

Laplacian Blending using Image Pyramids is a very good intro to working and thinking in frequencies, and Deconvolution is a neat trick.

You tasks for this assignment are:

    Perform Histogram Equalization on the given input image.
    Perform Low-Pass, High-Pass and Deconvolution on the given input image.
    Perform Laplacian Blending on the two input images (blend them together).
    
    
Your goal is to create 2 panoramas:

    Using homographies and perspective warping on a common plane (3 images).

    Using cylindrical warping (many images).

In both options you should:

    Read in the images: input1.jpg, input2.jpg, input3.jpg
    [Apply cylindrical wrapping if needed]
    Calculate the transformation (homography for projective; affine for cylindrical) between each
    Transform input2 and input3 to the plane of input1, and produce output.png
