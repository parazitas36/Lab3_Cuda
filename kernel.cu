#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "picojson.h"
#include <stdio.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

const int THREADS = 8;
const int WORD_SIZE = 10;
const int RESULT_SIZE = WORD_SIZE * 2;

int* intArr;
double* doubleArr;
char* charArr;
char* resultsArr;
int count = 0;

void Read() {
	std::cout << "Pasirinkit failą: ";
	int fileNr;
	std::cin >> fileNr;
	std::string FILENAME = "IFK9-2_GrosasL_L3_dat_";

	switch (fileNr) {
		case (2):
			FILENAME += "2.json";
			break;
		case(3):
			FILENAME += "3.json";
			break;
		default:
			FILENAME += "1.json";
			break;
	}

	std::cout << FILENAME << std::endl;
	std::ifstream dataFile(FILENAME);

	if (!dataFile) {
		std::cout << "Neatidare";
	}
	
	std::string json;
	std::string line;
	json = "";

	while (dataFile >> line) {
		json += line;
	}

	picojson::value v;
	picojson::parse(v, json);
	picojson::array arr = v.get("players").get<picojson::array>();
	count = arr.capacity();

	intArr = new int[count];
	doubleArr = new double[count];
	charArr = (char*)calloc(count * WORD_SIZE, sizeof(char));
	resultsArr = (char*)calloc(count * RESULT_SIZE, sizeof(char));
	int index = 0;

	for (picojson::array::iterator iter = arr.begin(); iter != arr.end(); ++iter) {
		double intArrDbl = (*iter).get("pts").get<double>();
		double dbl = (*iter).get("fgPct").get<double>();
		std::string name = (*iter).get("name").get<std::string>();
		char *carr = new char[name.length()];
		std::strcpy(carr, name.c_str());

		for (int i = 0; i < WORD_SIZE; i++) {
			if (i < name.length()) {
				charArr[WORD_SIZE * index + i] = carr[i];
			}
			else {
				break;
			}
		}

		intArr[index] = (int)intArrDbl;
		doubleArr[index] = dbl;
		index++;
	}
}


__global__ void filterGPU(int* d_int, double* d_double, char* d_chars, int* startIndices, int* endIndices, char* d_results, int *d_count, int *d_resultCount) {
	char* word = new char[WORD_SIZE];
	int threadID = threadIdx.x;
	int nameLength;
	char letter;

	for (int j = startIndices[threadID]; j < endIndices[threadID]; j++) {
		word = new char[WORD_SIZE];
		nameLength = 0;
		for (int i = 0; i < WORD_SIZE; i++) {
			letter = d_chars[j * WORD_SIZE + i];
			if (letter != NULL) {
				word[i] = letter;
				nameLength++;
			}
			else { break; }
		}

		char* result = new char[RESULT_SIZE];
		double compValue = d_double[j] * d_int[j];
		int whole = (int)compValue;
		
		if (compValue - (double)whole == 0) {
			char grade;

			if (whole > 4000) { grade = 'A'; }
			else if (whole > 3000) { grade = 'B'; }
			else if (whole > 2000) { grade = 'C'; }
			else if (whole > 1000) { grade = 'D'; }
			else { grade = 'F'; }

			printf("Thread:%d Name: %s Pts: %d Double: %f Comp:%f \n", threadIdx.x, word, d_int[j], d_double[j], compValue);
			for (int i = 0; i < nameLength; i++) {
				result[i] = word[i];
			}
			result[nameLength] = '-';
			result[nameLength + 1] = grade;
			for (int i = nameLength + 2; i < RESULT_SIZE; i++) {
				result[i] = ' ';
			}

			int index = atomicAdd(d_resultCount, 1);
			int position = index * RESULT_SIZE;

			for (int i = position; i < position + RESULT_SIZE; i++) {
				d_results[i] = result[i - position];
			}
		}
	}
}


int main()
{
	int* d_int;
	double* d_double;
	char* d_chars;
	char* d_results;
	int* startIndices = new int[THREADS];
	int* endIndices = new int[THREADS];
	int* d_startIndices;
	int* d_endIndices;
	int* d_count;
	int* d_resultCount;
	int resultCount = 0;

	// Failo nuskaitymas
	Read();
	
	// Duomenu paskirstymas
	int chunkSize = count / THREADS;
	int overFlow = count % THREADS;
	int tempOverFlow = count % THREADS;
	int offSet;
	int startIndex, endIndex;
	for (int i = 0; i < THREADS; i++) {
		if (i < overFlow) {
			offSet = overFlow - tempOverFlow;
			startIndex = i * chunkSize + offSet;
			endIndex = startIndex + chunkSize + 1;
			std::cout << "Thread:" << i << " Start=" << startIndex << " End=" << endIndex << std::endl;
			tempOverFlow--;
			startIndices[i] = startIndex;
			endIndices[i] = endIndex;
		}
		else {
			startIndex = i * chunkSize + overFlow;
			endIndex = startIndex + chunkSize;
			std::cout << "Thread:" << i << " Start=" << startIndex << " End=" << endIndex << std::endl;
			startIndices[i] = startIndex;
			endIndices[i] = endIndex;
		}
	}


	cudaMalloc((void**)&d_count, sizeof(int));
	cudaMalloc((void**)&d_resultCount, sizeof(int));
	cudaMalloc(&d_int, sizeof(int) * count);
	cudaMalloc(&d_double, sizeof(double) * count);
	cudaMalloc(&d_chars, sizeof(char) * WORD_SIZE * count);
	cudaMalloc(&d_results, sizeof(char) * RESULT_SIZE * count);
	cudaMalloc(&d_startIndices, sizeof(int) * THREADS);
	cudaMalloc(&d_endIndices, sizeof(int) * THREADS);

	cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_resultCount, &resultCount, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_int, intArr, sizeof(int) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_double, doubleArr, sizeof(double) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_chars, charArr, sizeof(char) * WORD_SIZE * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_results, resultsArr, sizeof(char) * RESULT_SIZE * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_startIndices, startIndices, sizeof(int) * THREADS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_endIndices, endIndices, sizeof(int) * THREADS, cudaMemcpyHostToDevice);

	filterGPU<<< 1, THREADS >>>(d_int, d_double, d_chars, d_startIndices, d_endIndices, d_results, d_count, d_resultCount);

	cudaDeviceSynchronize();

	cudaMemcpy(resultsArr, d_results, sizeof(char) * count * RESULT_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(&resultCount, d_resultCount, sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < resultCount * RESULT_SIZE; i++) {
		if (i % RESULT_SIZE == 0) {
			std::cout << "\nNR: " << (i/RESULT_SIZE)+1 << " ";
		}
		std::cout << resultsArr[i];
	}

	// Isvalyti atminti
	cudaFree(d_int);
	cudaFree(d_double);
	cudaFree(d_chars);
	cudaFree(d_results);
	cudaFree(d_startIndices);
	cudaFree(d_endIndices);
	cudaFree(d_count);
	cudaFree(d_resultCount);

	delete[] intArr;
	delete[] doubleArr;
	delete[] charArr;
	delete[] startIndices;
	delete[] endIndices;

	return 0;
}
