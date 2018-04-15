//#include "stdafx.h"//header for Visual Studio
#include <opencv2/core/core.hpp>//header for OpenCV core
#include <opencv2/highgui/highgui.hpp>//header for OpenCV UI
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>//header for c++ IO

#include <amp.h>
#include <amp_math.h>
#include <amp_graphics.h>
#include <thread>

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;

typedef std::chrono::steady_clock the_clock;

//set opencv and c++ namespaces
using namespace cv;
using namespace std;
using namespace concurrency;

#define TS 10
#define TX 480
#define TY 850

void convert_to_8UC1(Mat &img) {
	Mat channels[3];
	split(img, channels);
	img = channels[0];
	img.convertTo(img, CV_8UC1);
}


void edge(Mat* img) {

	int kernelX[3][3] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	int kernelY[3][3] = {
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
	};

	Mat edges;

	edges.create(img->rows, img->cols, 0);

	for (int x = 1; x < img->rows - 1; x++) {
		for (int y = 1; y < img->cols - 1; y++) {

			//construct matrix from surrounding pixels
			float x_total = 0; float y_total = 0;

			uchar pixel_matrix[3][3] = {
				img->at<uchar>(x - 1,y - 1),img->at<uchar>(x + 0,y - 1),img->at<uchar>(x + 1,y - 1),
				img->at<uchar>(x - 1,y + 0),img->at<uchar>(x + 0,y + 0),img->at<uchar>(x + 1,y + 0),
				img->at<uchar>(x - 1,y + 1),img->at<uchar>(x + 0,y + 1),img->at<uchar>(x + 1,y + 1)
			};

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					x_total += pixel_matrix[k_x][k_y] * kernelX[k_x][k_y];
					y_total += pixel_matrix[k_x][k_y] * kernelY[k_x][k_y];
				}
			}

			float intensity = sqrt((x_total * x_total) + (y_total * y_total));

			intensity /= 1141; //makes temp proportional to rough max value of convolution
			intensity *= 255; //converts proportion to intensity value

			edges.at<uchar>(x, y) = (int)intensity;

		}
	}

	*img = edges;

	return;

}

void threaded_edge(Mat* img) {

	int kernelX[3][3] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	int kernelY[3][3] = {
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
	};

	int cores = std::thread::hardware_concurrency();
	int seg_size = img->rows / cores;
	Mat edges;

	edges.create(img->rows, img->cols, 0);

	std::vector<std::thread> threads;

	for (int i = 0; i < cores; i++) {
		int x_start = seg_size * i; int x_end = (seg_size * i) + seg_size;
		if (x_start == 0) { x_start++; } if (x_end == img->rows) { x_end--; }
		threads.push_back(std::thread([=, &edges] () {
			for (int x = x_start; x < x_end - 1; x++) {
				for (int y = 1; y < img->cols - 1; y++) {

					//construct matrix from surrounding pixels
					float x_total = 0; float y_total = 0;

					uchar pixel_matrix[3][3] = {
						img->at<uchar>(x - 1,y - 1),img->at<uchar>(x + 0,y - 1),img->at<uchar>(x + 1,y - 1),
						img->at<uchar>(x - 1,y + 0),img->at<uchar>(x + 0,y + 0),img->at<uchar>(x + 1,y + 0),
						img->at<uchar>(x - 1,y + 1),img->at<uchar>(x + 0,y + 1),img->at<uchar>(x + 1,y + 1)
					};

					for (int k_x = 0; k_x < 3; k_x++) {
						for (int k_y = 0; k_y < 3; k_y++) {
							x_total += pixel_matrix[k_x][k_y] * kernelX[k_x][k_y];
							y_total += pixel_matrix[k_x][k_y] * kernelY[k_x][k_y];
						}
					}

					float intensity = sqrt((x_total * x_total) + (y_total * y_total));

					intensity /= 1141; //makes temp proportional to rough max value of convolution
					intensity *= 255; //converts proportion to intensity value

					edges.at<uchar>(x, y) = (int)intensity;

				}
		}

		}));
	}

	for (int i = 0; i < threads.size(); i++) { threads[i].join(); }

	*img = edges;

	return;
}

void edge_detect_AMP(Mat& source_img, Mat& dest_img) {
	int rows = source_img.rows;
	int cols = source_img.cols;

	concurrency::extent<2> source_imgSize(rows, cols);
	int bits = 8;

	const unsigned int nBytes = source_imgSize.size() * bits / 8;

	// Source Data
	graphics::texture<unsigned int, 2> texDataS(source_imgSize, source_img.data, nBytes, bits);
	graphics::texture_view<const unsigned int, 2> texS(texDataS);

	// Result data
	graphics::texture<unsigned int, 2> texDataD(source_imgSize, bits);
	graphics::texture_view<unsigned int, 2> texD(texDataD);


	parallel_for_each(source_imgSize, [=, &texDataS](concurrency::index<2> idx) restrict(amp)
	{
		int kernelX[3][3] = {
			-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1
		};

		int kernelY[3][3] = {
			-1, -2, -1,
			0, 0, 0,
			1, 2, 1
		};

		int val = texS(idx);
		val = texS(idx[0], idx[1]);
		// Do your source_img processing work here.


		//construct matrix from surrounding pixels
		float x_total = 0; float y_total = 0;

		float pixel_matrix[3][3] = {
			texS(idx[0] - 1, idx[1] - 1),texS(idx[0], idx[1] - 1),texS(idx[0] + 1, idx[1] - 1),
			texS(idx[0] - 1, idx[1] + 0),texS(idx[0] + 0, idx[1] + 0),texS(idx[0] + 1, idx[1] + 0),
			texS(idx[0] - 1, idx[1] + 1),texS(idx[0] + 0, idx[1] + 1),texS(idx[0] + 1, idx[1] + 1)
		};

		for (int k_x = 0; k_x < 3; k_x++) {
			for (int k_y = 0; k_y < 3; k_y++) {
				x_total += pixel_matrix[k_x][k_y] * kernelX[k_x][k_y];
				y_total += pixel_matrix[k_x][k_y] * kernelY[k_x][k_y];
			}
		}

		float temp = concurrency::fast_math::sqrt((x_total * x_total) + (y_total * y_total));

		temp /= 1141;

		temp *= 255;

		int result = temp;


		texD.set(idx, result);
	});

	// Don't copy async here as you need the result immediately.
	concurrency::graphics::copy(texDataD, dest_img.data, nBytes);

	return;

}

void tiled_edge_detect(Mat& source_img, Mat& dest_img) {
	int rows = source_img.rows;
	int cols = source_img.cols;

	concurrency::extent<2> source_imgSize(rows, cols);
	int bits = 8;

	const unsigned int nBytes = source_imgSize.size() * bits / 8;

	// Source Data
	graphics::texture<unsigned int, 2> texDataS(source_imgSize, source_img.data, nBytes, bits);
	graphics::texture_view<const unsigned int, 2> texS(texDataS);

	// Result data
	graphics::texture<unsigned int, 2> texDataX(source_imgSize, bits);
	graphics::texture<unsigned int, 2> texDataY(source_imgSize, bits);
	graphics::texture<unsigned int, 2> texDataR(source_imgSize, bits);

	graphics::texture_view<unsigned int, 2> tex_X(texDataX);
	graphics::texture_view<unsigned int, 2> tex_Y(texDataY);
	graphics::texture_view<unsigned int, 2> tex_R(texDataR);



	try {
		parallel_for_each(tex_Y.extent.tile<TX, 1>(), [=, &texDataS](concurrency::tiled_index<TX, 1> idx) restrict(amp) {

			const int kernelY[3][3] = {
				-1, -2, -1,
				0, 0, 0,
				1, 2, 1
			};

			//construct matrix from surrounding pixels
			tile_static int pixels[TX][3];

			int loc_x = idx.local[0], loc_y = idx.local[1] + 1;
			if (loc_y != 0) {
				pixels[loc_x][loc_y - 1] = texS(idx.global[0], idx.global[1] + 0);
			}
			if (loc_y != TY) {
				pixels[loc_x][loc_y + 1] = texS(idx.global[0], idx.global[1] + 2);
			}

			pixels[loc_x][loc_y] = texS(idx.global);

			idx.barrier.wait();

			float y_total = 0;

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					y_total += pixels[k_x + loc_x][k_y + loc_y] * kernelY[k_x][k_y];
				}
			}

			float intensity = concurrency::fast_math::sqrt((y_total * y_total));

			intensity /= 1020;

			intensity *= 255;


			tex_Y.set(idx, intensity);
		});

		parallel_for_each(tex_X.extent.tile<1, TY>(), [=, &texDataS](concurrency::tiled_index<1, TY> idx) restrict(amp) {

			const int kernelX[3][3] = {
				-1, 0, 1,
				-2, 0, 2,
				-1, 0, 1
			};

			//construct matrix from surrounding pixels
			tile_static int pixels[3][TY];

			int loc_x = idx.local[0] + 1, loc_y = idx.local[1];

			pixels[loc_x][loc_y] = texS(idx.global);
			if (loc_x != 0) {
				pixels[loc_x - 1][loc_y] = texS(idx.global[0] + 0, idx.global[1]);
			}
			if (loc_x != TX) {
				pixels[loc_x + 1][loc_y] = texS(idx.global[0] + 2, idx.global[1]);
			}

			idx.barrier.wait();

			float x_total = 0;

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					x_total += pixels[k_x + loc_x][k_y + loc_y] * kernelX[k_x][k_y];
				}
			}

			float intensity = concurrency::fast_math::sqrt((x_total * x_total));

			intensity /= 1020;

			intensity *= 255;

			tex_X.set(idx, intensity);
		});

		parallel_for_each(source_imgSize, [=](concurrency::index<2> idx) restrict(amp)
		{
			float temp = concurrency::fast_math::sqrt((tex_X(idx) * tex_X(idx)) + (tex_Y(idx) * tex_Y(idx)));
			tex_R.set(idx, temp);

		});

		concurrency::graphics::copy(texDataR, dest_img.data, nBytes);
	}
	catch (const std::exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}

	return;
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main()
{

	VideoCapture cap("robots.avi");

	if (!cap.isOpened()) {
		std::cout << "Video not opened" << std::endl;
	}

	Mat frame, sample;


	//namedWindow("frame", CV_WINDOW_AUTOSIZE);

	namedWindow("frame", CV_WINDOW_NORMAL);
	setWindowProperty("frame", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

	double rate = cap.get(CV_CAP_PROP_FPS);

	int delay = 1000 / rate;

	float time_elapsed = 0.f;
	int frames = 0;

	while (true)
	{

		the_clock::time_point start = the_clock::now();
		if (!cap.read(frame)) {
			break;
		}

		cvtColor(frame, frame, CV_BGR2GRAY);
		//convert_to_8UC1(frame);


		type2str(frame.type());
		frame.convertTo(frame, CV_8UC1);

		//convert_to_8UC1(frame);

		sample = frame;
		
		//edge_detect_AMP(frame, sample);
		//tiled_edge_detect(frame, sample);
		
		//edge(&frame);
		threaded_edge(&frame);

		imshow("frame", frame);

		the_clock::time_point end = the_clock::now();
		frames++;


		// Compute the difference between the two times in milliseconds
		auto time_taken = duration_cast<milliseconds>(end - start).count();

		time_elapsed += time_taken;


		if (time_elapsed >= 1000)
		{
			std::cout << "FPS: " << frames << endl;

			time_elapsed = 0; frames = 0;

		}

		if (waitKey(delay) >= 0) { break; }
	}

	cap.release();

	Mat image;//create mat variable for our image
	image = imread("test2.jpg", CV_8UC1); // read image.  1 : read as rgb, 0 : read as grayscale

	Mat image2;//create mat variable for our image

	image2 = imread("test2.jpg", CV_8UC1); // read image.  1 : read as rgb, 0 : read as grayscale

	the_clock::time_point start = the_clock::now();

	std::cout << type2str(image.type()) << endl;

	std::cout << type2str(image2.type()) << endl;

	//imshow("Edge", image);
	waitKey();

	edge_detect_AMP(image, image2);
	//tiled_edge_detect(&image, &image2);

	//edge(&image);

	image2 = image;
	the_clock::time_point end = the_clock::now();

	// Compute the difference between the two times in milliseconds
	auto time_taken = duration_cast<milliseconds>(end - start).count();
	std::cout << "Time taken: " << time_taken << " ms." << endl;

	if (!image.data) // check whether the image is found or not
	{
		std::cout << "Image is not found. Please write the file name and path location correctly." << endl;
	}

	namedWindow("Edge", CV_WINDOW_NORMAL);
	setWindowProperty("Edge", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow("Edge", image2);

	waitKey(0);// imshow will show image once the program hit this "waitKey".
	return 0;
}