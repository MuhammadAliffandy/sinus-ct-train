# Logbook Penelitian: Sinus CT Segmentation

Logbook ini mencatat riwayat pekerjaan, kendala, dan keputusan arsitektur yang diambil selama proses pelatihan model segmentasi sinus.

---

### 16 September 2026 - Setup Pipeline nnU-Net & Troubleshooting Awal

| **Aspek** | **Detail** |
| :--- | :--- |
| **Pekerjaan yang Dilakukan** | Melakukan *refactoring* pada skrip `run_pipeline.sh` dengan menambahkan argumen dinamis (`--dataset`, `--fold`, `--trainer`) untuk otomasi pelatihan dan *ensemble*. Memperbarui `evaluate_metrics.py`. Memulai pelatihan Fold 0 dengan pelatih khusus 250 epochs (`nnUNetTrainer_250epochs`). |
| **Kendala/Masalah** | Terminal tampak berhenti total (*stuck*) di awal Epoch 0. Ditemukan adanya potensi *multiprocessing hang* dan tabrakan memori di server DGX H100 akibat *Just-In-Time (JIT) Compilation* dari PyTorch. |
| **Solusi/Tindak Lanjut** | Mematikan kompilasi paksa (`export nnUNet_compile=f`) dan membatasi pembagian tugas *thread* ke CPU (`OMP_NUM_THREADS=1` dan `nnUNet_n_proc_DA=0`) untuk memastikan pelatihan berjalan lancar dan stabil di server DGX. |
| **Dokumentasi (Link/Ref)** | `run_pipeline.sh`, `evaluate_metrics.py`, `sinus_ct_pipeline_architecture.md` |

---

### 17 September 2026 - Evaluasi Fold 0 & Fitur Resume Training

| **Aspek** | **Detail** |
| :--- | :--- |
| **Pekerjaan yang Dilakukan** | Pelatihan Fold 0 selesai dengan skor akhir *Mean Validation Dice* ~73.4%. Memberikan skrip *monitoring real-time* menggunakan `watch` dan `tail` untuk melacak log dari berbagai *fold* secara otomatis. |
| **Kendala/Masalah** | Pelatihan Fold 1 terhenti dan mati secara diam-diam (*silent crash*) di Epoch 178 akibat koneksi SSH yang terputus (SIGHUP) karena proses tidak dibungkus dengan `nohup` atau `tmux`. |
| **Solusi/Tindak Lanjut** | Memodifikasi `run_pipeline.sh` dengan menyisipkan fitur Resume (`--continue` / `-c`). Fitur ini memungkinkan nnU-Net untuk mencari `checkpoint_latest.pth` dan melanjutkan pelatihan dari Epoch 177 tanpa harus mengulang dari awal (Epoch 0). |
| **Dokumentasi (Link/Ref)** | `run_pipeline.sh` (Updated) |

---

### 18 September 2026 - Evaluasi Performa & Peralihan (Pivot) ke SAM-Med3D

| **Aspek** | **Detail** |
| :--- | :--- |
| **Pekerjaan yang Dilakukan** | Menganalisis hasil Fold 1 yang tertahan di skor Dice ~71.4% (Anterior) dan 71.5% (Posterior). Berdiskusi mengenai strategi menembus batas akurasi >80% (SOTA). |
| **Kendala/Masalah** | Menggunakan 250 epochs menyebabkan model mengalami *underfitting* (tidak mendapat porsi penurunan *learning rate* yang matang). Namun, menggunakan standar 1000 epochs untuk 5 *Fold* memakan waktu komputasi yang sangat lama di server DGX (berhari-hari). |
| **Solusi/Tindak Lanjut** | Memutuskan untuk beralih (*pivot*) haluan arsitektur menuju **SAM-Med3D** (*Foundation Model*). Karena model ini hanya butuh *fine-tuning*, waktunya jauh lebih singkat. Membuat skrip `prepare_sam_dataset.py` untuk mengonversi label multi-kelas NIfTI menjadi label biner tunggal sesuai syarat mutlak input SAM-Med3D. Menyusun langkah-langkah instalasi SAM-Med3D di DGX. |
| **Dokumentasi (Link/Ref)** | `prepare_sam_dataset.py`, `implementation_plan.md`, `walkthrough.md`, `task.md` |

---

### 21 September 2026 - Penyelesaian Tahap 1 & Pause Proyek

| **Aspek** | **Detail** |
| :--- | :--- |
| **Pekerjaan yang Dilakukan** | Melakukan *troubleshooting* pada *pipeline* SAM-Med3D (menangani isu `CUDA Out of Memory` dengan menurunkan *batch size* ke 1, dan membuat *auto-patch* untuk kompatibilitas tensor *float* di kalkulasi *Dice Loss* MONAI). |
| **Kendala/Masalah** | Kebutuhan mendesak untuk beralih *(switch)* ke proyek lain sehingga eksperimen SAM-Med3D yang sedang/telah berjalan perlu dihentikan sementara atau direkapitulasi. |
| **Solusi/Tindak Lanjut** | Menyimpan (merekap) semua *training logs* dari nnU-Net dan SAM-Med3D sebagai bukti pengerjaan Tahap 1. Proyek `sinus-ct-train` di-*pause* sementara dengan infrastruktur yang sudah sepenuhnya siap untuk dilanjutkan kapan saja. |
| **Dokumentasi (Link/Ref)** | `run_sam_pipeline.sh` (Updated) |

---
