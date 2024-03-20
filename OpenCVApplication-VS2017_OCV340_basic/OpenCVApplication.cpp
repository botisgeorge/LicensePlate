#include "stdafx.h"
#include "common.h"

//----------------------------------------------------------PROIECT-------------------------------------------------------------

Mat grayscale(Mat img) {
	Mat img2(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b var = img.at< Vec3b>(i, j);
			uchar var2 = (var[0] + var[1] + var[2]) / 3;
			img2.at<uchar>(i, j) = var2;
		}
	}
	return img2;
}

Mat MybilateralFilter(const Mat& src, int d, double sigmaColor, double sigmaSpace) {
	CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);

	Mat result = Mat::zeros(src.size(), src.type());

	int halfD = d / 2;
	int height = src.rows;
	int width = src.cols;

	Mat spatialKernel = Mat::zeros(d, d, CV_64F);
	double spatialSigma = sigmaSpace;
	double spatialCoefficient = -0.5 / (spatialSigma * spatialSigma);
	for (int i = -halfD; i <= halfD; ++i) {
		for (int j = -halfD; j <= halfD; ++j) {
			double distanceSquared = i * i + j * j;
			spatialKernel.at<double>(i + halfD, j + halfD) = exp(distanceSquared * spatialCoefficient);
		}
	}

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (src.type() == CV_8UC1) {
				double intensitySum = 0;
				double weightSum = 0;

				for (int i = -halfD; i <= halfD; ++i) {
					for (int j = -halfD; j <= halfD; ++j) {
						int neighborX = borderInterpolate(x + j, width, BORDER_REFLECT);
						int neighborY = borderInterpolate(y + i, height, BORDER_REFLECT);
						double spatialWeight = spatialKernel.at<double>(i + halfD, j + halfD);
						double intensityDiff = src.at<uchar>(neighborY, neighborX) - src.at<uchar>(y, x);
						double colorWeight = exp(-0.5 * intensityDiff * intensityDiff / (sigmaColor * sigmaColor));
						double weight = spatialWeight * colorWeight;

						intensitySum += src.at<uchar>(neighborY, neighborX) * weight;
						weightSum += weight;
					}
				}

				result.at<uchar>(y, x) = static_cast<uchar>(intensitySum / weightSum);
			}
			else if (src.type() == CV_8UC3) {
				Vec3d intensitySum(0, 0, 0);
				double weightSum = 0;

				for (int i = -halfD; i <= halfD; ++i) {
					for (int j = -halfD; j <= halfD; ++j) {
						int neighborX = borderInterpolate(x + j, width, BORDER_REFLECT);
						int neighborY = borderInterpolate(y + i, height, BORDER_REFLECT);
						double spatialWeight = spatialKernel.at<double>(i + halfD, j + halfD);

						Vec3b intensityDiff = src.at<Vec3b>(neighborY, neighborX) - src.at<Vec3b>(y, x);
						double colorWeight = exp(-0.5 * (intensityDiff.dot(intensityDiff)) / (sigmaColor * sigmaColor));
						double weight = spatialWeight * colorWeight;

						intensitySum += src.at<Vec3b>(neighborY, neighborX) * weight;
						weightSum += weight;
					}
				}

				result.at<Vec3b>(y, x) = intensitySum / weightSum;
			}
		}
	}

	return result;
}

Mat cannyEdgeDetection(const Mat& src, double lowerThreshold, double upperThreshold) {
	CV_Assert(src.type() == CV_8UC1);

	Mat blurred;
	GaussianBlur(src, blurred, Size(5, 5), 0);

	Mat gradientX, gradientY;
	Sobel(blurred, gradientX, CV_64F, 1, 0);
	Sobel(blurred, gradientY, CV_64F, 0, 1);

	Mat magnitude, angle;
	cartToPolar(gradientX, gradientY, magnitude, angle, true);

	Mat nonMaxSuppressed = Mat::zeros(src.size(), CV_8UC1);

	int height = src.rows;
	int width = src.cols;

	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			double angleValue = angle.at<double>(y, x);

			double q = magnitude.at<double>(y, x + 1);
			double r = magnitude.at<double>(y, x - 1);
			double p = magnitude.at<double>(y + 1, x);
			double s = magnitude.at<double>(y - 1, x);

			if ((angleValue < 22.5 && angleValue >= 0) || (angleValue >= 157.5 && angleValue < 202.5) || (angleValue >= 337.5 && angleValue <= 360)) {
				if (magnitude.at<double>(y, x) > q && magnitude.at<double>(y, x) > r) {
					nonMaxSuppressed.at<uchar>(y, x) = magnitude.at<double>(y, x);
				}
			}
			else if ((angleValue >= 22.5 && angleValue < 67.5) || (angleValue >= 202.5 && angleValue < 247.5)) {
				if (magnitude.at<double>(y, x) > p && magnitude.at<double>(y, x) > s) {
					nonMaxSuppressed.at<uchar>(y, x) = magnitude.at<double>(y, x);
				}
			}
			else if ((angleValue >= 67.5 && angleValue < 112.5) || (angleValue >= 247.5 && angleValue < 292.5)) {
				if (magnitude.at<double>(y, x) > q && magnitude.at<double>(y, x) > s) {
					nonMaxSuppressed.at<uchar>(y, x) = magnitude.at<double>(y, x);
				}
			}
			else if ((angleValue >= 112.5 && angleValue < 157.5) || (angleValue >= 292.5 && angleValue < 337.5)) {
				if (magnitude.at<double>(y, x) > p && magnitude.at<double>(y, x) > r) {
					nonMaxSuppressed.at<uchar>(y, x) = magnitude.at<double>(y, x);
				}
			}
		}
	}

	Mat edgeMap = Mat::zeros(src.size(), CV_8UC1);

	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			uchar currentMagnitude = nonMaxSuppressed.at<uchar>(y, x);

			if (currentMagnitude >= upperThreshold) {
				edgeMap.at<uchar>(y, x) = 255;
			}
			else if (currentMagnitude >= lowerThreshold && currentMagnitude < upperThreshold) {
				if (nonMaxSuppressed.at<uchar>(y + 1, x) >= upperThreshold || nonMaxSuppressed.at<uchar>(y - 1, x) >= upperThreshold ||
					nonMaxSuppressed.at<uchar>(y, x + 1) >= upperThreshold || nonMaxSuppressed.at<uchar>(y, x - 1) >= upperThreshold ||
					nonMaxSuppressed.at<uchar>(y + 1, x + 1) >= upperThreshold || nonMaxSuppressed.at<uchar>(y - 1, x - 1) >= upperThreshold ||
					nonMaxSuppressed.at<uchar>(y - 1, x + 1) >= upperThreshold || nonMaxSuppressed.at<uchar>(y + 1, x - 1) >= upperThreshold) {
					edgeMap.at<uchar>(y, x) = 255;
				}
			}
		}
	}

	return edgeMap;
}

void MyfindContours(const Mat& src, std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, int mode, int method) {
	CV_Assert(src.type() == CV_8UC1);

	contours.clear();
	hierarchy.clear();

	Mat srcCopy = src.clone();
	Mat contoursMap = Mat::zeros(src.size(), CV_8UC1);

	int height = srcCopy.rows;
	int width = srcCopy.cols;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (srcCopy.at<uchar>(y, x) == 255 && contoursMap.at<uchar>(y, x) == 0) {
				std::vector<Point> contour;
				Point startPoint(x, y);
				Point currentPoint = startPoint;

				do {
					contour.push_back(currentPoint);
					contoursMap.at<uchar>(currentPoint) = 255;

					Point neighborPoints[8] = {
						Point(currentPoint.x - 1, currentPoint.y),
						Point(currentPoint.x + 1, currentPoint.y),
						Point(currentPoint.x, currentPoint.y - 1),
						Point(currentPoint.x, currentPoint.y + 1),
						Point(currentPoint.x - 1, currentPoint.y - 1),
						Point(currentPoint.x - 1, currentPoint.y + 1),
						Point(currentPoint.x + 1, currentPoint.y - 1),
						Point(currentPoint.x + 1, currentPoint.y + 1)
					};

					for (int i = 0; i < 8; ++i) {
						Point neighborPoint = neighborPoints[i];

						if (neighborPoint.x >= 0 && neighborPoint.x < width && neighborPoint.y >= 0 && neighborPoint.y < height &&
							srcCopy.at<uchar>(neighborPoint) == 255 && contoursMap.at<uchar>(neighborPoint) == 0) {
							currentPoint = neighborPoint;
							break;
						}
					}
				} while (currentPoint != startPoint);

				contours.push_back(contour);
				hierarchy.push_back(Vec4i(-1, -1, -1, -1));
			}
		}
	}
}

void licensePlate() {
	char fname[MAX_PATH];
	Mat image;
	while (openFileDlg(fname)) {
		image = imread(fname, 1);
		// resizing the image
		int newWidth = 1000;
		resize(image, image, Size(newWidth, image.rows * newWidth / image.cols));

		// transformare in grayscale
		Mat gray;
		gray = grayscale(image);

		// noise remove with bilateral filter
		Mat filteredImg;
		//								  d  sigma color / sigma space
		bilateralFilter(gray, filteredImg, 11, 17, 17);
		//filteredImg = MybilateralFilter(gray, 11, 17, 17);

		// find edges using canny
		int lower = 170;
		int upper = 200;
		Mat edged;
		Canny(filteredImg, edged, lower, upper);
		//edged = cannyEdgeDetection(filteredImg, lower, upper);
		// dilate the edges to find rectangles easier
		Mat dilated;
		dilate(edged, dilated, Mat(), Point(-1, -1), 1);

		// find contours
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;
		findContours(dilated.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
		//MyfindContours(dilated.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

		// sort the countours decreasingly
		sort(contours.begin(), contours.end(), [](const std::vector<Point>& a, const std::vector<Point>& b) {
			return contourArea(a, false) > contourArea(b, false);
			});

		Mat output = image.clone();

		// loop over the contours to find the best possible approximate contour of the number plate
		for (const auto& contour : contours) {
			double peri = arcLength(contour, true);
			double epsilon = 0.01 * peri;

			std::vector<Point> approx;
			approxPolyDP(contour, approx, epsilon, true);

			if (approx.size() <= 5) {
				// check if the contour is at least 1000 pixels big and has a reasonable aspect ratio
				double area = contourArea(approx);
				Rect boundingRect = cv::boundingRect(approx);
				double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
				if (area >= 1000 && (aspectRatio >= 0.2 && aspectRatio <= 5)) {
					//if yes then put it inside a blue box
					polylines(output, approx, true, Scalar(255, 255, 0), 5);
					break;
				}
			}
		}

		// Display the original image
		imshow("Input Image", image);
		// Display Grayscale image
		imshow("Gray scale Image", gray);
		// Display Filtered image
		imshow("After Applying Bilateral Filter", filteredImg);
		// Display Canny Image
		imshow("After Canny Edges", edged);
		// Display Dilated Image
		imshow("After Dilation", dilated);
		// Display the output image
		newWidth = 750;
		resize(output, output, Size(newWidth, image.rows * newWidth / image.cols));
		imshow("Output", output);

		waitKey(0); // Wait for user input before closing the displayed images
	}
}



int main()
{
	licensePlate();
	return 0;
}