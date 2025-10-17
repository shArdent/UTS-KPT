import asyncio
import aiohttp
import aiofiles
import os
import time
import random
from tqdm.asyncio import tqdm_asyncio

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 410 gambar dengan resolusi acak antara 200–800 piksel
urls = [
    f"https://picsum.photos/seed/{i}/{random.randint(200,800)}/{random.randint(200,800)}"
    for i in range(500)
]

SEM_LIMIT = 50  # batas koneksi paralel

async def download_image(session, sem, url):
    async with sem:
        filename = os.path.join(SAVE_DIR, f"{url.split('/')[-3]}_{url.split('/')[-2]}x{url.split('/')[-1]}.jpg")
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    async with aiofiles.open(filename, "wb") as f:
                        await f.write(await resp.read())
                    return True
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
        return False

async def main():
    start = time.perf_counter()
    sem = asyncio.Semaphore(SEM_LIMIT)
    async with aiohttp.ClientSession() as session:
        results = await tqdm_asyncio.gather(
            *[download_image(session, sem, url) for url in urls],
            desc="Downloading random-resolution images"
        )
    elapsed = time.perf_counter() - start
    success = sum(results)
    print(f"\n✅ {success}/{len(urls)} images downloaded in {elapsed:.2f}s")
    print(f"📊 Throughput: {success/elapsed:.2f} images/sec")

if __name__ == "__main__":
    asyncio.run(main())
