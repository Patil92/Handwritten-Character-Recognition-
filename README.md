# Handwritten-Character-Recognition-
Implementation of Handwritten Character Recognition using OCR Technique and KNN  Algorithm.

1.	Introduction

1.1 Introduction 
	Handwriting recognition has been one of the most fascinating and challenging research areas in field of image processing and pattern recognition in the recent years. It contributes immensely to the advancement of automation process and improves the interface between man and machine in numerous applications. Several research works have been focusing on new techniques and methods that would reduce the processing time while providing higher recognition accuracy. In general, handwriting recognition is classified into two types as off-line and on-line handwriting recognition methods. In the off-line recognition, the writing is usually captured optically by a scanner and the completed writing is available as an image. But, in the on-line system the two dimensional coordinates of successive points are represented as a function of time and the order of strokes made by the writer are also available. The on-line methods have been shown to be superior to their off-line counter parts in recognizing handwritten characters due to the temporal information available with the former. However, in the off-line systems, the neural networks have been successfully used to achieve comparably high recognition accuracy levels. Several applications including mail sorting, bank processing, document reading and postal address recognition require offline handwriting recognition systems. As a result, the off-line handwriting recognition continues to be an active area for research towards exploring the newer techniques that would improve recognition accuracy. In this paper a neural based off-line handwritten character recognition system, without feature extraction is proposed. The pre-processed image is segmented into individual characters. Each character is resized into 20x20 pixels and these pixels are used to train a feed forward back propagation neural network employed for performing classification and recognition tasks. Extensive simulation studies show that the recognition system provides good recognition accuracy. 

1.2	Problem Statement
Optical character recognition is a part of biometric system. Since 1929, number of character recognition systems have been develop and are use for even commercial purpose also. Several applications including mail sorting, bank processing, document reading and postal address recognition, bank checking process require off-line handwriting systems. Working in Postal service need us to decode and deliver something like 30 million handwritten envelopes every single day. The challenges are to do mail-sorting that ensure all those millions of letters reach their destinations. When it comes to processing more human kinds of information, it is a hard task for computer to do because we have to “communicate” to them through relatively crude devices such as keyboards and mice so they can figure out what we want them to do. 
	In the India and most of other countries, bank cheques are pre-printed with the account number and the check number in special ink and format; as such, these two numeric fields can be easily read and processed using computerized techniques. However, the amount fields on a filled-in check is usually read by human eyes, and involves significant time and cost, especially when one considers that over 50 million checks are processed per annum in the India alone. In order to overcome this, we have to develop algorithms that can make computer understand the handwriting and to do the bank check task. By referring to the stated problem, we have chosen the handwritten recognition as a solution to these problem. But, to implement this task we also face any problem that are:

1. Machine simulation of human functions has been a very challenging Research field since the advent of digital computers
2. HCR is a challenging problem since there is a variation of same character due to the change of fonts and sizes.
3. The differences in font types and sizes make the recognition task difficult and resulting the recognition of character process becomes not accurate.

1.3 Objectives
The objectives of the project are: 
1. To improve the techniques or method for HCR from variation of font and size.
2. To study and improve the algorithm in order to minimize the error when analysing the extracted features of the handwriting image to achieved a greater accuracy.
3. To analyse the performance of Artificial Neural Network.

2. Literature Survey
2.1 Previous Research	
	An early notable attempt in the area of character recognition research is by Grim dale in 1959. The origin of a great deal of research work in the early sixties was based on an approach known as analysis-by-synthesis method suggested by Eden in 1968. The great importance of Eden's work was that he formally proved that all handwritten characters are formed by a finite number of schematic features, a point that was implicitly included in previous works. This notion was later used in all methods in syntactic (structural) approaches of character recognition. K. Gaurav, Bhatia P. K. Et al, this paper deals with the various pre-processing techniques involved in the character recognition with different kind of images ranges from a simple handwritten form based documents and documents containing coloured and complex background and varied intensities. In this, different pre-processing techniques like skew detection and correction, image enhancement techniques of contrast stretching, binarization, noise removal techniques, normalization and segmentation, morphological processing techniques are discussed. It was concluded that using a single technique for pre-processing, we can’t completely process the image. However, even after applying all the said techniques might not possible to achieve the full accuracy in a pre-processing system. Salvador España-Boquera et al, in this paper hybrid Hidden Markov Model (HMM) model is proposed for recognizing unconstrained offline handwritten texts. In this, the structural part of the optical model has been modelled with Markov chains, and a Multilayer Perceptron is used to estimate the emission probabilities. In this paper, different techniques are applied to remove slope and slant from handwritten text and to normalize the size of text images with supervised learning methods. The key features of this recognition system were to develop a system having high accuracy in pre-processing and recognition, which are both based on ANNs. In a modified quadratic classifier based scheme to recognize the offline handwritten numerals of six popular Indian scripts is proposed. Multilayer perceptron has been used for recognizing Handwritten English characters. The features are extracted from Boundary tracing and their Fourier Descriptors. The character is identified by analysing its shape and comparing its features that distinguish each character. Also an analysis has been carried out to determine the number of hidden layer nodes to achieve high performance of the back propagation network. A recognition accuracy of 94% has been reported for handwritten characters with less training time.


3. Requirements
3.1 Software Requirements
•	Python 2.7/3.5
		Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. ... Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance.
•	Anaconda IDE
		Anaconda (Python distribution) Anaconda is a free and open source distribution of the Python and R programming languages for data science and machine learning related applications (large-scale data processing, predictive analytics, scientific computing), that aims to simplify package management and deployment.
•	Opencv 2
		OpenCV-Python. OpenCV-Python is a library of Python bindings designed to solve computer vision problems. ... OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax. All the OpenCV array structures are converted to and from Numpy arrays.
•	Numpy ,matplotlib, image module
		Numpy is a highly optimized library for numerical operations. It gives a MATLAB-style syntax. All the OpenCV array structures are converted to-and-from Numpy arrays. Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension Numpy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+.
•	Hand written character Dataset
		A data set (or dataset) is a collection of data. Most commonly a data set corresponds to the contents of a single database table, or a single statistical data matrix, where every column of the table represents a particular variable, and each row corresponds to a given member of the data set in question.


4. Implementation
4.1 KNN Algorithm 
	KNN algorithm is one of the simplest classification algorithm and it is one of the most used learning algorithms. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point. When we say a technique is non-parametric, it means that it does not make any assumptions on the underlying data distribution. In other words, the model structure is determined from the data. If you think about it, it’s pretty useful, because in the “real world”, most of the data does not obey the typical theoretical assumptions made (as in linear regression models, for example). Therefore, KNN could and probably should be one of the first choices for a classification study when there is little or no prior knowledge about the distribution data.
KNN can be used for classification — the output is a class membership (predicts a class — a discrete value).
	An object is classified by a majority vote of its neighbours, with the object being assigned to the class most common among its k nearest neighbours. It can also be used for regression — output is the value for the object (predicts continuous values). This value is the average (or median) of the values of its k nearest neighbours.

 
Figure 4.1 KNN Algorithm
	KNN can be used for both classification and regression predictive problems. However, it is more widely used in classification problems in the industry. To evaluate any technique we generally look at 3 important aspects:
1. Ease to interpret output
2. Calculation time
3. Predictive Power
KNN algorithm fairs across all parameters of considerations. It is commonly used for its easy of interpretation and low calculation time. 
 
Figure 4.2 KNN Classification
	Given that all the 6 training observation remain constant, with a given K value we can make boundaries of each class. These boundaries will segregate RC from GS. The same way, let’s try to see the effect of value “K” on the class boundaries. Following are the different boundaries separating the two classes with different values of K.

 
Figure 4.3 KNN Classification
	In this section, the proposed recognition system is described. A typical handwriting recognition system consists of pre-processing, segmentation, classification and post processing stages. The general schematic diagram of the recognition system is shown in Fig.1.The proposed method which does not include feature extraction stage is shown in Fig.2.


4.1.1 Image acquisition
	In Image acquisition, the recognition system acquires a scanned image as an input image. The image should have a specific format such as JPEG, BMT etc. This image is acquired through a scanner, digital camera or any other suitable digital input device.
4.1.2 Pre-processing
	The pre-processing is a series of operations performed on the scanned input image. It essentially enhances the image rendering it suitable for segmentation. The various tasks performed on the image in pre-processing stage are shown in Fig.3 Binarization process converts a gray scale image into a binary image using global thresholding technique. Dilation of edges in the binarized image is done using sobel technique, dilation the image and filling the holes present in it are the operations performed in the last two stages to produce the pre-processed image suitable for segmentation.

	In the above Flow chart figure 4.5 we explained the working flow of handwritten character recognition. In first we are going to read our image from disk and then convert into grayscale. This grayscale image is blurred using GuassianBlur technique which blur the image i.e. unwanted information are removed from the image. Then the canny edges of the image are detected using cv2 canny function. Then the contours present in the image are extracted one by one.as the image extracted more contours we want only important contours so we filter out the contours. The bounded image is cropped at particular pixels for only important features. Then we binarized the image into 20x20 pixel. As the comparison of 1D array is faster than multidimensional we convert the array into 1x400 pixel array. And the outputted array is given to classifier to detect the digit.
4.1.3 Segmentation
	In the segmentation stage, an image of sequence of characters is decomposed into sub-images of individual character. In the proposed system, the pre-processed input image is segmented into isolated characters by assigning a number to each character using a labelling process. This labelling provides information about number of characters in the image. Each individual character is uniformly resized into 20x20 pixels.

4.1.4 Classification and Recognition
The classification stage is the decision making part of the recognition system. A feed forward back propagation neural network is used in this work for classifying and recognizing the handwritten characters. The 400 pixels derived from the resized character in the segmentation stage form the input to the classifier. The neural classifier consists of two hidden layers besides an input layer and an output layer as shown in. The hidden layers use log sigmoid activation function and the output layer is a competitive layer as one of the characters is required to be identified at any point in time. The total number of neurons in the output layer is 10 (0 t0 9 here) as the proposed system is designed to recognize numbers.

4.1.5 About Digits dataset:
•	The digits.png image contains 500 samples of each numeral (0-9).
•	Total of 5000 samples of data 
•	Each individual character has dimensions: 20 x 20 pixels
•	Is grayscale with black backgrounds

4.1.6 Preparing our Dataset
•	500 samples of each digits with 5 rows of 100 samples 
•	Each character is grayscale 20x20 pixels
•	We use numpy to arrange the data in this format:
	50 x 100 x 20 x 20
•	We then split the training dataset into 2 segments and flatten our 20x20 array.
a.	Training set – 50% of data
b.	Test Set – 50% of data – we use a test set to evaluate our modal
c.	Each dataset is then flattened, meaning we turn the 20x20 pixels array into a flat 1x400. Each row of 20 pixels is simply appended into one long column.
•	We then assign labels to both training & test datasets (i.e. 0,1,…9)

5. Results and Conclusion
We have proposed and developed a scheme for recognizing hand written Numbers. We have tested our experiment overall Numbers with several Hand writing styles. Experimental results shown that the machine has successfully recognized the alphabets with the average accuracy of more than 75%, which significant and may be acceptable in some applications. The machine found less accurate to classify similar Numbers and in future this misclassification of the similar patterns may improve and further a similar experiment can be tested over a large data set and with some other optimized networks parameters to improve the accuracy of the machine. The pixel values derived from the resized characters of the segmentation stage have been directly used for training the modal. As a result, the proposed system will be less complex compared to the offline methods using feature extraction techniques. Of the several networks modal architectures used for classifying the characters, the one with two hidden layers each having 400  has been found to yield the highest recognition accuracy of 91.76%. The handwritten recognition system described in this paper will find potential applications in handwritten name recognition, document reading, conversion of any handwritten document into structural text form and postal address recognition.


6. Future Scope
The proposed algorithms used for segmentation of handwritten Number recognition can be extended further for recognition of other Indian scripts.
• The proposed algorithms of segmentation can be modified further to improve Accuracy of segmentation.
• New features can be added to improve the accuracy of recognition.
• These algorithms can be tried on large database of handwritten text.
• There is a need to develop the standard database for recognition of handwritten text.
• The proposed work can be extended to work on degraded text or broken characters.
• Recognition of digits in the text, half characters and compound characters can be done to improve the word recognition rate.



6. Applications

•	Used in Banking sectors for verification of checks
•	Used in Self driving cars to detect the sign boards
•	Used in data interpretation of sign boards
•	Used to convert handwritten language into digital such in forms of pdfs, word documents etc.


7. References

  https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
  https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea
  http://ijcsit.com/docs/Volume%207/vol7issue1/ijcsit2016070101.pdf
  https://ieeexplore.ieee.org/document/5402783
  https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea



