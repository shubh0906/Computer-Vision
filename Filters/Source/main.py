# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   #split the input image into its bgr channels
   ycc = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCR_CB)
   
   channels = cv2.split(ycc);
   hist, bins = numpy.histogram(channels[0], 256, [0,256])
   cdf = hist.cumsum()
  # cdf_normalized = cdf * float(hist.max()) / cdf.max()
   cdf_m = numpy.ma.masked_equal(cdf, 0)
   cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
   cdf = numpy.ma.filled(cdf_m, 0).astype('uint8');
   channels[0] = cdf[channels[0]]
   print channels[0]
   img_out = cv2.merge(channels);
   img_out = cv2.cvtColor(img_out, cv2.COLOR_YCR_CB2BGR)   
#   img_out = img_in
   return True, img_out

   '''# Write histogram equalization here
   img_out = img_in # Histogram equalization result 
   return True, img_out'''
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):

   #img_in = cv2.cvtColor(img_in, 0) 
   dft = cv2.dft(numpy.float32(img_in), flags = cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = numpy.fft.fftshift(dft)

   magnitude_spectrum = 20*numpy.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

   rows, cols = img_in.shape
   crows, ccol = rows/2, cols/2

   #Create mask first: LPF mask has center pixel 1 and rest is 0
   mask = numpy.zeros((rows,cols,2), numpy.uint8)
   mask[crows-10:crows+10, ccol-10:ccol+10] = 1

   # Apply mask and Inverse DFT
   fshift = dft_shift*mask   
   f_ishift = numpy.fft.ifftshift(fshift)
   img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
   img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

   return True, img_back
   '''# Write low pass filter here
   img_out = img_in # Low pass filter result
	
   return True, img_out'''

def high_pass_filter(img_in):

   dft = cv2.dft(numpy.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = numpy.fft.fftshift(dft)
   rows, cols = img_in.shape
   crow, ccol = rows / 2, cols / 2
   dft_shift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
   f_ishift = numpy.fft.ifftshift(dft_shift)
   img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
   img_out = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
   return True, img_out

   '''# Write high pass filter here
   img_out = img_in # High pass filter result
   
   return True, img_out'''
   
def deconvolution(img_in):
   
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T

   def ft(img_in, newsize=None):
      dft = numpy.fft.fft2(numpy.float32(img_in),newsize)
      return numpy.fft.fftshift(dft)

   def ift(shift):
      f_ishift = numpy.fft.ifftshift(shift)
      img_back = numpy.fft.ifft2(f_ishift)
      return numpy.abs(img_back)

   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
   gkf = ft(gk, (img_in.shape[0], img_in.shape[1])) # so we can multiple easily
   imconvf = imf /gkf
   
   # now for example we can reconstruct the blurred image from its FT
   blurred = ift(imconvf)
   blurred = cv2.multiply(blurred, 255)
   return True, blurred
   '''# Write deconvolution codes here
   img_out = img_in # Deconvolution result
   
   return True, img_out'''

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   #Gaussian pyramid of img_in1
   img_in1=img_in1[:,:img_in1.shape[0]]
   img_in2=img_in2[:img_in1.shape[0],:img_in1.shape[0]]
   levels = 5
   gp_img_in1=[img_in1.copy()]
   for i in range(1, levels):
      gp_img_in1.append(cv2.pyrDown(gp_img_in1[i - 1]))

   #Gaussian pyramid of img_in2
   gp_img_in2=[img_in2.copy()]
   for i in range(1, levels):
      gp_img_in2.append(cv2.pyrDown(gp_img_in2[i - 1]))

   # Generate the inverse Laplacian Pyramid for img_in1
   lp_img_in1 = [gp_img_in1[-1]]
   for i in range(levels - 1, 0, -1):
      laplacian = cv2.subtract(gp_img_in1[i - 1], cv2.pyrUp(gp_img_in1[i]))
      lp_img_in1.append(laplacian)

   # Generate the inverse Laplacian Pyramid for img_in2
   lp_img_in2 = [gp_img_in2[-1]]
   for i in range(levels - 1, 0, -1):
      laplacian = cv2.subtract(gp_img_in2[i - 1], cv2.pyrUp(gp_img_in2[i]))
      lp_img_in2.append(laplacian)

   # Add the left and right halves of the Laplacian images in each level
   laplacianPyramidComb = []
   for l_img_in1, l_img_in2 in zip(lp_img_in1, lp_img_in2):
      rows, cols, dpt = l_img_in1.shape
      laplacianComb = numpy.hstack((l_img_in1[:, 0:cols / 2], l_img_in2[:, cols / 2:]))
      laplacianPyramidComb.append(laplacianComb)

   # Reconstruct the image from the Laplacian pyramid
   imgComb = laplacianPyramidComb[0]
   for i in range(1, levels):
      imgComb = cv2.add(cv2.pyrUp(imgComb), laplacianPyramidComb[i])
      img_out = imgComb # Blending result
   
   return True, img_out

   '''# Write laplacian pyramid blending codes here
   img_out = img_in1 # Blending result
   
   return True, img_out'''

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
