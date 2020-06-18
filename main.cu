#include <iostream>
#include <algorithm> 
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "device_launch_parameters.h"
#include <vector>
#include <cuda_profiler_api.h>
#include <D:/EE817/cuPT/stb_image_write.h>

using namespace std;

#define NSTREAM 4
#define BLOCKSIZE 16

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, int offset, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j + offset >= max_y)) return;
	int pixel_index = j*max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index + offset * max_x, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int offset, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j + offset >= max_y)) return;
	int pixel_index = j*max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	float u, v;
	ray r;
	for (int s = 0; s < ns / 8; s++) {
		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);

		u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		v = float(j + offset + curand_uniform(&local_rand_state)) / float(max_y);
		r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RND;
				vec3 center(a + RND, 0.2, b + RND);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2,
						new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0.f, 1.f, 0.f), 1.0f, new dielectric(1.5f));
		d_list[i++] = new sphere(vec3(-4.f, 1.f, 0.f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
		d_list[i++] = new sphere(vec3(4.f, 1.f, 0.f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
		*rand_state = local_rand_state;
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0f; (lookfrom - lookat).length();
		float aperture = 0.1f;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20.0,
			float(nx) / float(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

unsigned int clamp(int p)
{
	return p < 0 ? p : p > 0xff ? 0xff : p;
}

int main() {
	int nx = 1200;
	int ny = 800;
	int ns = 1 << 5;
	int tx = BLOCKSIZE;
	int ty = BLOCKSIZE;

	std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);

	// allocate an image buffer
	vec3 *h_buffer, *d_buffer;
	checkCudaErrors(cudaHostAlloc((void **)&h_buffer, fb_size, cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc((void **)&d_buffer, fb_size));
	memset(h_buffer, 0, fb_size);
	//vec3 *fb;
	//checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
	curandState *d_rand_state_w;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state_w, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> >(d_rand_state_w);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	hitable **d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world << <1, 1 >> >(d_list, d_world, d_camera, nx, ny, d_rand_state_w);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render our buffer
	int iElem = (ny + NSTREAM - 1) / NSTREAM;
	dim3 grid((nx + tx - 1) / tx, (iElem + ty - 1) / ty);
	dim3 block(tx, ty);

	cudaStream_t *streams = (cudaStream_t *)malloc(NSTREAM * sizeof(cudaStream_t));
	for (int i = 0; i < NSTREAM; i++)
	{
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	cudaProfilerStart();
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < NSTREAM; i++)
	{
		int iOffset = i * iElem;
		render_init << <grid, block, 0, streams[i] >> >(nx, ny, iOffset, &d_rand_state[iOffset * nx]);
		//checkCudaErrors(cudaMemcpyAsync(&d_buffer[iOffset * nx], &h_buffer[iOffset * nx], min(iElem, ny - iOffset) * nx * sizeof(vec3), cudaMemcpyHostToDevice, streams[i]));
		render << <grid, block, 0, streams[i] >> >(&d_buffer[iOffset * nx], nx, ny, iOffset, ns, d_camera, d_world, &d_rand_state[iOffset * nx]);
		checkCudaErrors(cudaMemcpyAsync(&h_buffer[iOffset * nx], &d_buffer[iOffset * nx], min(iElem, ny - iOffset) * nx * sizeof(vec3), cudaMemcpyDeviceToHost, streams[i]));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float timer_seconds;
	checkCudaErrors(cudaEventElapsedTime(&timer_seconds, start, stop));

	std::cerr << "gpu took " << timer_seconds / 1000 << " seconds.\n";
	cudaProfilerStop();

	// Output FB as Image
	std::vector<unsigned char> pix(nx * ny * 3);
	for (int j = ny - 1; j >= 0; --j) {
		unsigned char *dst = &pix[0] + (3 * nx*(ny - j - 1));
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;

			int ir = int(255.99*h_buffer[pixel_index].r());
			*dst++ = static_cast<unsigned char>(clamp(ir));
			int ig = int(255.99*h_buffer[pixel_index].g());
			*dst++ = static_cast<unsigned char>(clamp(ig));
			int ib = int(255.99*h_buffer[pixel_index].b());
			*dst++ = static_cast<unsigned char>(clamp(ib));
		}
	}

	const char* filename = "./out_cuda.png";
	std::string suffix;
	std::string fn(filename);
	if (fn.length() > 4) {
		suffix = fn.substr(fn.length() - 4);
	}
	if (suffix != ".png")
	{
		throw std::string("Only support .png format: ") + filename;
	}
	if (!stbi_write_png(filename, (int)nx, (int)ny, 3, &pix[0], nx * 3 * sizeof(unsigned char)))
	{
		throw std::string("Failed to write image: ") + filename;
	}


	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> >(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_buffer));

	cudaDeviceReset();
}
