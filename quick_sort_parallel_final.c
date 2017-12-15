#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define NMAX 100000000

// A utility function to swap two elements
void swap(double *a, double *b){
    double t = *a;
    *a = *b;
    *b = t;
}

// Parallel Partition
int parallelPartition (double arr[], int low, int high){
	// choose pivot
	double pivot = arr[high];
	int n = high - low + 1;
	
	// < and > boolean arrays
	int *lessThan = malloc(n * sizeof(int));
	int *greaterThan = malloc(n * sizeof(int));
	double *placeholder = malloc(n * sizeof(double));
	int i = low;
	
	// populate less than and greater than arrays
	#pragma omp parallel for schedule(guided)
	for (i = 0; i < n; i++){
		placeholder[i] = arr[i + low];
		if (arr[i + low] <= pivot) {
			lessThan[i] = 1;
			greaterThan[i] = 0;
		} else {
			lessThan[i] = 0;
			greaterThan[i] = 1;
		}
	}
	int j;
	int step;
	
	// Number of steps for prefix sum
	int steps = ceil(log(n) / log(2));

	// Run prefix sum on less than and greater than arrays
	for (i = 0; i < steps; i++) {
		step = pow(2, i + 1);
		#pragma omp parallel for schedule(guided)
		for (j = 0; j < n; j += step) {
			int target = j + step - 1;
			int prev = j + step / 2 - 1;
			if (target < n) {
				lessThan[target] = lessThan[prev] + lessThan[target];
				greaterThan[target] = greaterThan[prev] + greaterThan[target];
			}
		}
	
	}

	// downwards step
	for (i = steps - 2; i >= 0; i--) {
		step = pow(2, i + 1);
		#pragma omp parallel for schedule(guided)
		for (j = 0; j < n; j += step) {
			int target = j + step  + step/2 - 1;
			int prev = j + step - 1;
			if (target < n) {
				lessThan[target] = lessThan[prev] + lessThan[target];
				greaterThan[target] = greaterThan[prev] + greaterThan[target];
			}
		}
	}
	
	// swap elements using corresponding prefix summed array values as indeces
	#pragma omp parallel for schedule(guided)
	for (i = low; i <= high; ++i) {
		if (placeholder[i - low] <= pivot)
			arr[low + lessThan[i - low] - 1] = placeholder[i - low];
		else
			arr[low + lessThan[n - 1] + greaterThan[i - low] - 1] = placeholder[i - low];
	}

	int part = low + lessThan[n-1] - 1;
	free(lessThan);
	free(greaterThan);
	free(placeholder);
	return part;
}

// function to find the partition double 
int partition (double arr[], int low, int high){
    double pivot = arr[high]; 
    int i = (low - 1); 
 
    int j;

    for (j = low; j <= high- 1; j++){
        if (arr[j] <= pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// function to perform an insertion sort sequentially
void insertionSort(double arr[], int n)
{
    int i, j;
    double k;

    // loop to go through each of n elements in array
    for (i = 1; i < n; i++){
        k = arr[i];
        j = i-1;

        // loop backwards through already sorted elements until reaching spot for new one
        while (j >= 0 && arr[j] > k){
            arr[j+1] = arr[j];
            j = j-1;
        }

        // put the new element where it belongs in the sorted portion
        arr[j+1] = k;
    }    
}

// function to perform the recursive sequential quicksort
void quickSortSequential(double arr[], int low, int high){
    if (low < high)
    {
        // if array portion is small enough use a "fast" sorting algorithm
        if(high - low < 8){
            insertionSort(&arr[low], high - low + 1);
        }
        else{
            // partition the array
            int pi = partition(arr, low, high);

            quickSortSequential(arr, low, pi - 1);
            quickSortSequential(arr, pi + 1, high);
        }
    }
}

// function to perform the recursive parallel quicksort
void quickSortParallel(double arr[], int low, int high, int cutoff){
    if (low < high)
    {
        int pi;

        // cascading used at smaller chunk sizes
        // this number comes from trial and error, optimized for 16 threads
        if(high - low < 1900) {
            pi = partition(arr, low, high);

            quickSortSequential(arr, low, pi - 1);
            quickSortSequential(arr, pi + 1, high);
        }
        else{
            // partition. If num of elements in section is < .6 of total n, use seq partition
            if (high - low + 1 > cutoff)
                pi = parallelPartition(arr, low, high);
            else
                pi = partition(arr, low, high);

            #pragma omp task
            quickSortParallel(arr, low, pi - 1, cutoff);
            #pragma omp task
            quickSortParallel(arr, pi + 1, high, cutoff); 
        }
    }
}

// function to shuffle an array of doubles
void shuffleArray(double *array, int n){
    if (n > 1){
        int i;
        for (i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            double t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// function to print the array
void printArray(double *array, int n){
    int i;
    for(i = 0; i < n; i++){
        printf("%lf\n", array[i]);
    }
}

int main(int argc, char * argv[]){
    // how many doubles
    unsigned long long n;
    if (argc > 1) {
        n = atoll(argv[1]);
        n = (n > 0 && n <= NMAX) ? n : NMAX;
    }

    // make the array with n double values
    double *arr = malloc(n*sizeof(double));
    unsigned long long i;
    for(i = 0; i < n; i++){
        arr[i] = (double)i;
    }

    // randomly shuffle the array
    shuffleArray(arr, n);

	// compute cutoff for parallel vs sequential partition. .6 was found to provide optimal speedup.
	int cutoff = (int)(n * .6);
	
    // get start time
    double t1 = omp_get_wtime();

    // call quicksort
    // starts a parallel region - team of threads spawned
    #pragma omp parallel
    {
        // declare single so tasks are only spawned once
        // nowait due to uneven balance of partition
        #pragma omp single nowait
        {
            quickSortParallel(arr, 0, n-1, cutoff);
        }
    }

    // get end time and calculate total time
    double t2 = omp_get_wtime();
    printf("%llu,%lf,%d\n", n, t2-t1, omp_get_max_threads());

    // free up malloced space
    free(arr);

}