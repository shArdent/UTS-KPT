from mpi4py import MPI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image, ImageFilter
import numpy as np
import os, time, multiprocessing, argparse

# ---------------- IMAGE PROCESSING FUNCTION ----------------
def analyze_image(path, apply_filter=False):
    """Analisis satu gambar: mean warna, resolusi, filter opsional"""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    arr = np.array(img, dtype=np.float32)

    # Hitung mean warna (R,G,B)
    mean_rgb = arr.mean(axis=(0, 1))

    if apply_filter:
        img = img.filter(ImageFilter.BLUR)

    return {
        "filename": os.path.basename(path),
        "resolution": f"{width}x{height}",
        "mean_rgb": tuple(round(v, 2) for v in mean_rgb),
    }

# ---------------- HELPER FUNCTIONS ----------------
def list_image_paths(root_dir, limit=None):
    """Ambil semua gambar dari folder"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    if limit:
        return paths[:limit]
    return paths

def chunkify(lst, n):
    """Bagi list ke dalam n bagian"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# ---------------- MAIN PROGRAM ----------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--imdir", type=str, default="images", help="folder gambar")
    parser.add_argument("--num_threads", type=int, default=3)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--num_images", type=int, default=410)
    parser.add_argument("--sequential_time", type=float, default=None,
                        help="waktu eksekusi sekuensial (untuk hitung speedup)")
    args = parser.parse_args()

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Mulai timer total
    start_time = time.perf_counter()

    # MASTER rank
    if rank == 0:
        all_paths = list_image_paths(args.imdir, limit=args.num_images)
        print(f"[MASTER] Total gambar: {len(all_paths)}")
        chunks = chunkify(all_paths, size)
    else:
        chunks = None

    # Scatter ke semua rank
    my_paths = comm.scatter(chunks, root=0)
    print(f"[RANK {rank}] Menerima {len(my_paths)} gambar")

    # THREADS untuk prefetch
    with ThreadPoolExecutor(max_workers=args.num_threads) as tpool:
        prefetched = list(tpool.map(lambda p: p, my_paths))

    # PROCESSES untuk analisis gambar
    results = []
    with ProcessPoolExecutor(max_workers=args.num_procs) as ppool:
        futures = [ppool.submit(analyze_image, path, True) for path in prefetched]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[RANK {rank}] Error: {e}")

    # Gather hasil ke MASTER
    all_results = comm.gather(results, root=0)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # RANK 0: hitung metrik performa
    if rank == 0:
        flat_results = [item for sub in all_results for item in sub]
        num_images = len(flat_results)
        throughput = num_images / total_time

        # Hitung speedup & efisiensi
        if args.sequential_time:
            speedup = args.sequential_time / total_time
            efficiency = speedup / (size * args.num_procs)
        else:
            speedup = None
            efficiency = None

        print("\n===== HASIL EKSPERIMEN PARALLEL IMAGE PROCESSOR =====")
        print(f"Jumlah gambar       : {num_images}")
        print(f"Jumlah rank MPI     : {size}")
        print(f"Proses per rank     : {args.num_procs}")
        print(f"Thread per rank     : {args.num_threads}")
        print(f"Total proses aktif  : {size * args.num_procs}")
        print(f"Waktu eksekusi total: {total_time:.4f} detik")
        print(f"Throughput          : {throughput:.2f} gambar/detik")

        if speedup:
            print(f"Speedup             : {speedup:.2f}x")
            print(f"Efisiensi           : {efficiency*100:.2f}%")

        print("=====================================================\n")
        print("Contoh hasil analisis:")
        for r in flat_results[:5]:
            print(f"{r['filename']:<20} | {r['resolution']:<10} | RGB mean: {r['mean_rgb']}")
