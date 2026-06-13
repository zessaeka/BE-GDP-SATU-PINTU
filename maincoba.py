import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io
from io import BytesIO
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.image("income.png", use_container_width=True)
image_path = r'static/BACKGROUND InY (2).png'

def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error setting background: {e}")

set_background(image_path)

st.markdown(
    """
    <style>
    body {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab0"

# HALO (menu utama)
if st.session_state.active_tab == "tab0":
    st.header("Halo, GDPians~ Mau belajar apa hari ini?")
    st.info("Pilih menu belajar kamu di bawah, ya.")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    with col1:
        if st.button("Teori Pendapatan Nasional"):
            st.session_state.active_tab = "tab1"
    with col2:
        if st.button("Pendekatan Produksi"):
            st.session_state.active_tab = "tab3"
    with col3:
        if st.button("Pendekatan Pendapatan"):
            st.session_state.active_tab = "tab4"
    with col4:
        if st.button("Pendekatan Pengeluaran"):
            st.session_state.active_tab = "tab5"
    with col5:
        if st.button("Pendapatan Perkapita"):
            st.session_state.active_tab = "tab6"
    with col6:
        if st.button("Distribusi Pendapatan"):
            st.session_state.active_tab = "tab7"
    with col7:
        if st.button("Latihan Soal"):
            st.session_state.active_tab = "tab8"
    with col8:
        if st.button("Fakta-Fakta Menarik"):
            st.session_state.active_tab = "tab9"
    with col9:
        if st.button("Daftar Referensi"):
            st.session_state.active_tab = "tab10"  

# TEORI
elif st.session_state.active_tab == "tab1":
    st.header("Teori dan Konsep Pendapatan Nasional")
    st.markdown("""
    Pendapatan nasional merupakan total pendapatan yang diperoleh oleh seluruh pelaku ekonomi dalam suatu negara selama periode tertentu, biasanya satu tahun.
    Besarnya pendapatan nasional dipengaruhi oleh berbagai faktor, seperti ketersediaan faktor produksi, kualitas sumber daya manusia, teknologi,
    modal, stabilitas nasional, serta kebijakan yang diterapkan.
    Pendapatan nasional mencerminkan tingkat produksi barang dan jasa yang dihasilkan oleh suatu negara.""")
    st.divider()
    st.subheader("Pendekatan Perhitungan Pendapatan Nasional")
    st.write("Perhitungan pendapatan nasional dapat menggunakan tiga macam pendekatan, yaitu:")
    st.markdown("""
    - Pendekatan Produksi. Yaitu menghitung pendapatan nasional dengan menjumlahkan nilai tambah semua output yang dihasilkan.""")
    st.info("Rumusnya: Y = (P1 x Q1) + (P2 x Q2).... (Pn x Qn)")
    st.markdown("""
    - Pendekatan Pendapatan. Yaitu menghitung pendapatan nasional dengan menjumlahkan semua pendapatan yang diperoleh pelaku ekonomi, baik individu maupun perusahaan.""")
    st.info("Rumusnya: Y = r + w + i + p")
    st.markdown("""
    - Pendekatan Pengeluaran. Yaitu menghitung pendapatan nasional dengan menjumlahkan seluruh pengeluaran yang dilakukan oleh empat pelaku ekonomi utama (rumah tangga konsumen, perusahaan, pemerintah, dan masyarakat luar negeri) dalam suatu periode tertentu.
    """)
    st.info("Rumusnya:  Y = C + G + I + (X - M)")
    st.warning("Kita pelajari detail tiga macam pendekatan ini di menu belajar selanjutnya, ya!")
    st.divider()
    st.subheader("Manfaat Pendapatan Nasional")
    st.markdown("""
    - Memahami apakah struktur ekonomi suatu negara lebih dominan di sektor industri, agraris, atau jasa.
    - Membandingkan kemajuan ekonomi suatu negara dari waktu ke waktu serta antarnegara.
    - Memberikan pedoman kepada pemerintah untuk menyusun kebijakan pembangunan ekonomi nasional.
    - Menilai kinerja ekonomi suatu negara berdasarkan data pendapatan nasional.""")
    st.divider()            
    st.subheader("Konsep dan Urutan Komponen Pendapatan Nasional")
    st.write("1. Gross Domestic Product (GDP)")
    st.write("""GDP adalah total nilai pasar dari seluruh barang dan jasa jadi yang diproduksi di dalam batas wilayah geografis
    suatu negara selama periode tertentu (biasanya satu tahun). Teori GDP menganut asas teritorial atau wilayah.
    Artinya, faktor produksi milik siapapun (WNI maupun warga WNA) yang beroperasi di dalam negeri tersebut, nilainya mutlak dihitung ke dalam GDP.
    Nilainya masih kotor (Gross)""")
    st.write("""Supaya memudahkan kita dalam memahami, bayangkan kita punya restoran. Semua uang yang masuk dari penjualan makanan di dalam gedung restoran kita itu adalah GDP.
    kita tidak peduli apakah yang memasak itu kokinya WNI atau koki dari WNA, dan kita juga tidak peduli apakah pembelinya WNI atau WNA.
    Selama transaksi dan produksinya terjadi di dalam gedung restoran kita, semuanya dihitung sebagai omset kotor restoran.
    """)
    
    st.write("2. Gross National Product (GNP).")
    st.write("""GNP adalah total nilai pasar dari seluruh barang dan jasa jadi yang dihasilkan oleh faktor produksi yang dimiliki oleh
    warga negara suatu negara, tidak peduli di mana produksi itu terjadi secara geografis. Teori GNP menganut asas nasionalitas atau kewarganegaraan.
    Untuk mendapat nilai GNP, nilai GDP tadi harus disesuaikan dengan Pendapatan Neto terhadap Luar Negeri (pendapatan warga negara di luar negeri dikurangi pendapatan warga asing di dalam negeri).
    """)
    st.write("""Nah, tadi di GDP kita menghitung seluruh omset restoran tanpa melihat status kwarganegaraan. Pada konsep GNP, kita hanya menghitung omset yang dihasilkan oleh WNI saja, sehingga harus dikurangkan dengan pendapatan WNA.
    """)
    st.info("Rumusnya: GNP = GDP + Pendapatan WNI di Luar Negeri - Pendapatan WNA di Dalam Negeri")
    
    st.write("3. Net National Product (NNP).")
    st.write("""Dalam proses produksi, barang-barang modal seperti mesin, gedung, peralatan pasti mengalami keausan, penyusutan fungsi, atau kerusakan seiring berjalannya waktu.
    Secara teori, NNP diperoleh dengan cara mengurangi GNP dengan Penyusutan (Depresiasi) atau penggantian barang modal. NNP mencerminkan nilai bersih dari output nasional yang sebenarnya.
    """)
    st.write("""Setelah mengetahui total pendapatan murni dari WNI, kita sadar bahwa selama setahun blender di dapur ada yang rusak, kompor aus, dan piring banyak yang pecah. Kita tidak bisa menganggap semua uang yang dihasilkan sebagai keuntungan bersih
             karena kita harus menyisihkan sebagian uang untuk menyervis dan mengganti alat-alat masak yang rusak tersebut. Sisa uang setelah disisihkan untuk servis inilah yang disebut NNP. Sampai sini paham, kan?
    """)
    st.info("Rumusnya: NNP = GNP - Penyusutan")

    st.write("4. Net National Income (NNI).")
    st.write("""NNI adalah pendapatan bersih suatu negara yang dihitung dari NNP dikurangi pajak tidak langsung (seperti Pajak Pertambahan Nilai).
    NNI menunjukkan jumlah pendapatan yang diterima masyarakat sebagai balas jasa faktor produksi.""")
    st.write("""Omset bersih restoran tadi siap dibagikan nih ke pekerja. Tapi tunggu dulu, di dalam komponen harga makanan yang kita jual di pasar, ada titipan pajak dari pemerintah berupa Pajak Tidak Langsung yang menempel pada produk dan harus disetor ke negara,
    jadi itu bukan termasuk omset bersih restoran ya, jadi harus dikurangkan. Di sisi lain, kita juga mendapatkan bantuan subsidi gas dari pemerintah yang meringankan modalmu. Jadi, kita kurangi uang pajak itu dan masukkan modal subsidi tadi untuk melihat berapa nilai
    riil uang kas yang murni dihasilkan dari kerja keras pekerja restoran.
    """)
    st.info("Rumusnya: NNI = NNP - Pajak Tidak Langsung + Subsidi")
    
    st.write("5. Personal Income (PI).")
    st.write("""PI adalah bagian dari pendapatan nasional yang benar-benar diterima oleh setiap individu/orang per orang dalam masyarakat. Namun, tidak semua pendapatan nasional yaitu nilai NNI langsung mengalir ke kantong individu, 
    melainkan NNI harus dikurangi dengan bagian yang ditahan oleh institusi atau perusahaan seperti Laba Ditahan, Pajak Perseroan/Perusahaan, Iuran Asuransi, dan Iuran Jaminan Sosial. Kemudian ditambah dengan Transfer Payment (penerimaan yang bukan balas jasa produksi sekarang, melainkan jaminan sosial, pensiun, atau bansos dari pemerintah).
    """)
    st.write("""Omset bersih restoran kita sekarang mau dibagi-bagikan ke tiap-tiap pekerja, nih. Namun sebelum masuk dompet masing-masing, ada kesepakatan yakni sebagian uang harus disimpan di restoran untuk modal bisnis tahun depan, hal ini disebut Laba Ditahan, dan setiap orang wajib menyetor uang kas untuk iuran asuransi, misalnya Iuran Jaminan Sosial.
    Setelah proses potong iuran ini, barulah uang tersebut ditransfer ke rekening pribadi masing-masing anggota keluarga, uang inilah yang disebut Personal Income.
    """)
    st.info("Rumusnya: PI = (NNI + Transfer Payment) - (Laba ditahan + Iuran Asuransi + Pajak Perseroan + Jaminan Sosial)")
    
    st.write("6. Disposable  Income (DI).")
    st.write("""DI adalah pendapatan pribadi yang sudah bersih dari segala kewajiban hukum terhadap negara dan sepenuhnya siap digunakan untuk keperluan konsumsi barang/jasa,
    sedangkan sisanya dialokasikan menjadi tabungan/saving yang berputar kembali menjadi investasi. Secara teori, DI diperoleh dengan mengurangi PI dengan Pajak Langsung yaitu pajak yang wajib dibayar sendiri oleh wajib pajak dan tidak dapat dialihkan, seperti Pajak Penghasilan atau PPh.
    """)
    st.write("""Uang sudah masuk ke rekening pribadi pekerja kita, nih. Tapi pekerja belum boleh langsung memakainya untuk foya-foya atau belanja baju baru. Sebagai warga negara yang baik, pekerja harus membayar Pajak Penghasilan yang termasuk Pajak Langsung. Nah, setelah saldo rekening terpotong pajak, sisa uang ini barulah berstatus DI.
    Uang ini sudah 100% hak mutlak milik pekerja yang bebas dipakai untuk belanja kebutuhan masing-masing atau bisa dijadikan tabungan""")
    st.info("Rumusnya: DI = PI - Pajak Langsung")
    
    st.warning("Dari penjelasan tadi, konsep GDP hingga DI tadi saling terkait antara satu sama lain, ya. Sehingga, jika kita menemukan soal-soal menghitung PI misalnya, harus dihitung mulai dari GNP-NNP-NNI baru bisa mendapatkan nilai PI.")
    st.divider()
    st.subheader("Yuk, coba kerjakan latihan soal ini!")

    st.write("""
    Perhatikan data perekonomian suatu negara berikut (dalam miliar rupiah):

    - Gross Domestic Product: Rp500.000
    - Pendapatan WNA di dalam negeri: Rp45.000
    - Pendapatan WNI di luar negeri: Rp30.000
    - Depresiasi: Rp15.000
    - Pajak Tidak Langsung: Rp22.000
    - Subsidi: Rp7.000
    - Laba Ditahan: Rp8.000
    - Iuran Asuransi: Rp4.000
    - Pajak Perseroan: Rp3.000
    - Transfer Payment: Rp12.000
    - Pajak Langsung: Rp14.000

    Berdasarkan data di atas, besar Personal Income (PI) yang tepat adalah...
    """)

    jawaban_pi = st.radio(
        "Pilih jawaban yang benar:",
        (
            "A. Rp445.000 miliar",
            "B. Rp455.000 miliar",
            "C. Rp462.000 miliar",
            "D. Rp470.000 miliar",
            "E. Rp485.000 miliar"
        ),
        index=None,
        key="soal_pi"
    )

    if st.button("Periksa Jawaban", key="btn_pi"):

        if jawaban_pi is None:
            st.warning("Silakan pilih salah satu jawaban terlebih dahulu, ya!")

        if jawaban_pi == "B. Rp455.000 miliar":

            st.balloons()

            st.success("Jawaban kamu benar!")

            st.info("""
    Langkah Penyelesaian:

    1. Cari GNP

    GNP = GDP - Pendapatan WNA di DN + Pendapatan WNI di LN

    = 500.000 - 45.000 + 30.000

    = Rp485.000

    2. Cari NNP

    NNP = GNP - Penyusutan

    = 485.000 - 15.000

    = Rp470.000

    3. Cari NNI

    NNI = NNP - Pajak Tidak Langsung + Subsidi

    = 470.000 - 22.000 + 7.000

    = Rp455.000

    4. Cari PI

    Pengurang PI:
    - Laba Ditahan = 8.000
    - Iuran Asuransi = 4.000
    - Pajak Perseroan = 3.000

    Total Pengurang = 15.000

    PI = (NNI + Transfer Payment) - Pengurang

    = (455.000 + 12.000) - 15.000

    = Rp455.000 miliar
    """)
            st.warning("Masih semangat? Yuk, kita lanjut ke menu belajar selanjutnya!")

        else:

            st.error("Jawaban kamu masih kurang tepat!")

            if jawaban_pi == "A. Rp445.000 miliar":
                st.warning("""
    Kamu kemungkinan salah mengurangi komponen pengurang PI 
    atau lupa menambahkan transfer payment.
    """)

            elif jawaban_pi == "C. Rp462.000 miliar":
                st.warning("""
    Kamu kemungkinan sudah menambahkan transfer payment, 
    tetapi salah menghitung total pengurang PI.
    """)

            elif jawaban_pi == "D. Rp470.000 miliar":
                st.warning("""
    Kamu kemungkinan berhenti di tahap NNP 
    dan belum menghitung PI secara lengkap.
    """)

            elif jawaban_pi == "E. Rp485.000 miliar":
                st.warning("""
    Kamu kemungkinan hanya menghitung GNP 
    dan belum melanjutkan ke tahap berikutnya.
    """)

    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# PENDEKATAN PRODUKSI
elif st.session_state.active_tab == "tab3":
    st.header("Konsep Pendekatan Produksi")
    st.info("""
    Kamu bisa menggunakan menu ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Produksi. Selamat belajar, ya!
    """)

    st.write("""
    Sebelum mulai menghitung pendapatan nasional, kamu harus paham dahulu mengenai konsep Barang Antara (Intermediate Goods)
    dan Barang Final (Final Goods). Perlu diingat, bahwa yang kita hitung dalam pendekatan ini adalah penjumlahan nilai tambah dari total produksi seluruh sektor.
    Selain itu, juga harus bisa membedakan mana barang yang langsung dikonsumsi masyarakat dan mana yang masih akan diproses lagi.
             
    1. Barang Antara (Intermediate Goods).""")
    st.write("""Barang antara adalah barang yang diproduksi oleh suatu perusahaan, lalu dibeli oleh perusahaan lain untuk digunakan sebagai bahan baku atau komponen yang akan diproses lagi menjadi produk baru.
            Contohnya, ada sebuah toko roti yang membeli 50 kg telur dari peternak untuk dijadikan bahan baku membuat kue ulang tahun yang nantinya dijual. Nah, telur disini termasuk barang antara.
            Karena telur tersebut pasti akan dihancurkan, dicampur adonan, dan nilainya akan melebur ke harga kue ketika dijual nanti.""")
    
    st.write("""Tapi, tidak semua bahan mentah itu termasuk ke barang antara, lho. Contohnya saja buah-buahan atau sayuran segar.
            Banyak yang menganggap selalu jadi barang antara karena belum dimasak. Padahal kalau dibeli langsung oleh kita untuk dimakan, itu sudah jadi barang akhir.
            Lalu, mesin jahit yang dibeli oleh pabrik konveksi itu juga bukan termasuk barang antara, karena pabrik tekstil tidak mengubah bentuk mesin jahit itu untuk dicampur ke dalam baju.
            Mesin itu dipakai dalam jangka panjang, sehingga masuk sebagai barang akhir di kategori investasi perusahaan.""")
    
    st.write("""Jadi, nilai barang antara TIDAK boleh dimasukkan dalam perhitungan Pendapatan Nasional untuk menghindari double counting, karena nilainya nanti otomatis melekat pada harga barang akhir.""")
    st.write("""
    2. Barang Akhir (Final Goods).""")
    st.write("""Barang akhir adalah barang yang diproduksi dan dibeli oleh konsumen akhir (bisa masyarakat rumah tangga, pemerintah, atau perusahaan yang menggunakannya sebagai aset tetap, seperti mesin jahit tadi).
            Contohnya, Ibu kos membeli 1 kg telur di pasar untuk digoreng dan dimakan sebagai lauk sarapan anak-anak kos. Telur tersebut termasuk barang akhir, karena telur itu dibeli oleh konsumen akhir dan langsung habis dikonsumsi, tidak ada proses penjualan kembali setelahnya.""")
            
    st.write("""Nah, nilai barang akhir inilah yang WAJIB dimasukkan dalam perhitungan Pendapatan Nasional.          
    """)
    st.info("""Supaya memudahkan dalam memahami konsepnya, kita bisa pakai aturan emas ini, yaitu lihat status sebuah barang ditentukan oleh siapa yang membelinya dan untuk apa barang itu digunakan, bukan dari bentuk fisiknya.
            Jika barang itu masih mengalir ke pabrik/toko lain, berarti rantainya belum putus dan masih termasuk barang Antara. Tetapi, begitu barang itu masuk ke tas belanjaan konsumen rumah tangga, dan rantai produksinya putus di situ, maka disebut barang akhir.
            Bagaimana, mudah dipahami bukan?
            """)

    st.divider()

    st.header("Yuk, coba kita latihan sebentar di bawah ini!")
    st.write("Baca kasus berikut, lalu pilihlah barang yang disebut termasuk barang apa!")
    pertanyaan = {
        "Tepung dibeli pabrik roti untuk membuat roti": "Barang Antara",
        "Sepeda dibeli siswa untuk pergi ke sekolah": "Barang Akhir",
        "Mesin jahit dibeli pabrik konveksi": "Barang Akhir",
        "Gula dibeli perusahaan minuman untuk produksi": "Barang Antara",
        "Sayur dibeli ibu rumah tangga untuk dimasak": "Barang Akhir"
    }

    score = 0

    for soal, jawaban_benar in pertanyaan.items():
        pilihan = st.radio(
            soal,
            ["Barang Antara", "Barang Akhir"],
                index=None,
                key=soal
            )

    if pilihan == jawaban_benar:
                score += 1


    if st.button("Cek Jawaban"):
        st.success(f"Skor kamu: {score} dari {len(pertanyaan)}")

        if score == len(pertanyaan):
            st.balloons()
            st.success("Hebat! Semua jawaban benar.")
        elif score >= 3:
            st.info("Bagus! Kamu sudah cukup memahami konsepnya.")
        else:
            st.warning("Yuk belajar lagi supaya makin paham.")

    st.write("""
    Setelah kita memahami konsep barang antara dan barang akhir, barulah kita bisa menghitung nilai pendapatan nasional dengan mudah.
    Ingat, pada pendekatan produksi yang kita hitung adalah total dari nilai tambah produksi dari seluruh sektor.""")

    st.write("Perhatikan tabel di bawah ini!")
    st.image('produksi.png')

    st.write("""
    1. Apa itu Output Total?""")
    st.write("""Output Total adalah harga jual akhir suatu barang. Coba kita lihat data di atas pada kolom Nilai Penjualan (Output Total).
        Jika kita menjumlahkan semua Output Total tersebut, di sinilah terjadi kesalahan fatal bernama double counting atau penghitungan ganda.
        Mengapa? Karena di dalam harga kain yang Rp60.000 itu, sebenarnya sudah termasuk harga benang dan harga kapas. Kalau dijumlahkan semua, harga kapasnya akan terhitung berkali-kali.
        Sehingga nilainya tidak akurat.""")
    
    st.write("""
    2. Apa itu Nilai Tambah (Value Added)?""")
    st.write("""Nilai Tambah adalah selisih peningkatan harga karena adanya tenaga kerja dan proses produksi baru. Seperti tabel di atas, 
        Ketika pabrik benang membeli kapas seharga Rp10.000 lalu mengubahnya jadi benang seharga Rp25.000,
        pabrik itu tidak menciptakan uang Rp25.000 baru. Mereka hanya menambahkan nilai sebesar Rp15.000 ke kapas tersebut.
        Ketika tukang jahit mengubah kain Rp60.000 menjadi kemeja seharga Rp100.000, keahlian menjahitnya memberikan nilai tambah sebesar Rp40.000.
    """)
    
    st.write("""Kesimpulannya, dalam menghitung Pendapatan Nasional dengan pendekatan produksi, kita punya dua pilihan cara yang hasilnya akan sama persis, yaitu melihat harga barang akhir atau menjumlahkan nilai tambah.
    Hanya menghitung Harga Barang Akhir (Final Goods), yaitu harga kemeja di toko saja sebesar Rp100.000.
    Menjumlahkan Semua Nilai Tambah di tiap tahap: Rp10.000 + Rp15.000 + Rp35.000 + Rp40.000 = Rp100.000.
    """)
    st.divider()
    st.header("Waktunya latihan lagi! Yuk, coba kita kerjakan soal berikut!")

    st.markdown("""
    Diketahui data produksi suatu negara sebagai berikut:

    - Sektor perkebunan memproduksi gandum senilai Rp200 miliar.
    - Rp150 miliar dijual ke pabrik tepung.
    - Rp50 miliar langsung dijual ke masyarakat.
    - Pabrik tepung menghasilkan tepung terigu senilai Rp270 miliar.
    - Tepung dibeli pabrik roti.
    - Pabrik roti menghasilkan roti senilai Rp400 miliar.

    Berapakah total Pendapatan Nasional berdasarkan pendekatan nilai tambah?
    """)

    jawaban = st.radio(
        "Pilih jawaban yang benar:",
        (
            "A. Rp350 miliar",
            "B. Rp400 miliar",
            "C. Rp450 miliar",
            "D. Rp500 miliar",
            "E. Rp550 miliar"
        ),
        index=None,
    )

    if st.button("Cek Jawaban", key="nilai_tambah"):

        if jawaban == "C. Rp450 miliar":
            st.balloons()
            st.success("Jawaban kamu benar!")
            
            st.info("""
    Perhitungan nilai tambah:
    - Sektor perkebunan = Rp200 miliar
    - Pabrik tepung = Rp270 - Rp150 = Rp120 miliar
    - Pabrik roti = Rp400 - Rp270 = Rp130 miliar
    Total:
    200 + 120 + 130 = Rp450 miliar

    Atau kamu bisa perhatikan saja harga barang akhir, yaitu gandum yang dijual kepada konsumen senilai Rp50 miliar dan roti senilai Rp400 miliar.  
    Mudah bukan?
    """)
        else:
            st.error("Jawaban kamu masih kurang tepat. Coba perhatikan konsep nilai tambah lagi, ya!")
     
    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# PENDEKATAN PENDAPATAN
elif st.session_state.active_tab == "tab4":
    st.header("Konsep Pendekatan Pendapatan")
    st.info("""
    Kamu bisa menggunakan menu ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Pendapatan, ya. Selamat belajar, ya!
    """)

    st.write("""
    Sebelum kita menghitung menggunakan pendapatan ini, kita wajib memahami dahulu apa saja komponen yang dihitung dan masuk ke dalam perhitungan.
    Terdapat empat komponen yang masuk, yaitu Rent (Sewa), Wage (Upah), Interest (Bunga), dan Profit (Laba).
    Eh, terasa familiar bukan dengan istilah-istilah tersebut? Yap, benar. Empat macam balas jasa dari Rrumah Tangga Produksi kepada Rumah Tangga Konsumen atas faktor produksi dimasukkan semuanya ke perhitungan pendekatan
    pendapatan ini.
    Yuk, kita bahas lebih detail ya.
    """)
    
    st.write("1. Rent (Sewa)")
    st.write("""Rent/Sewa adalah pendapatan yang diterima oleh pemilik Faktor Produksi Alam / Tanah (Land).
    Perusahaan membutuhkan tempat untuk mendirikan pabrik, kantor, lahan pertanian, atau mengeksplorasi sumber daya alam.
    Yaitu dengan kata lain merupakan balas jasa karena pihak lain menyewa properti, lahan, atau kekayaan alam yang kita miliki.
    """)
    st.write("""Biasanya, pada soal-soal perhitungan pendekatan ini istilah lain yang muncul bisa Sewa Tanah, Sewa Kantor, Sewa Kendaraan, 
    atau Royalti. Jikalau nanti kita menemukan istilah-istilah tersebut di soal, jangan bingung ya, karena itu pasti termasuk ke dalam Sewa/Rent, yang intinya **Pendapatan Sewa** yang diterima.
    """)

    st.write("2. Wage (Upah)")
    st.write("""Wage/Upah adalah pendapatan yang diterima oleh pemilik Faktor Produksi Tenaga Kerja (Labor) karena telah menyumbangkan waktu, energi, dan pikirannya untuk membantu proses produksi.
    Upah adalah semua bentuk uang yang diterima karena seseorang berstatus sebagai pekerja/karyawan. Upah juga sering diistilahkan dengan Tunjangan-tunjangan, Bonus, Gaji, atau Kompensasi Kerja.
    """)
    st.write("Sering kali kita mengira Wage/Upah itu hanya gaji pokok. Padahal, semua bentuk tunjangan dan bonus yang melekat pada status sebagai pekerja harus dijumlahkan semuanya ke dalam komponen ini.")
    
    st.write("3. Interest (Bunga)")
    st.write("""Interest/Bunga adalah pendapatan yang diterima oleh pemilik Faktor Produksi Modal (Capital). Modal di sini tidak melulu berupa uang tunai, bisa juga berupa investasi peralatan atau dana yang dipinjamkan kepada perusahaan untuk operasional.
    Pada soal, biasanya diistilahkan juga sebagai Bunga Neto, Pendapatan Bunga, atau Bunga Obligasi.
    """)
    st.write("""Seringkali simbol bunga yaitu huruf i kecil (i) tertukar dengan simbol investment yaitu I besar (I) yang digunakan pada perhitungan pendapatan pengeluaran. Jangan bingung, ya.
    Interest/Bunga disimbolkan dengan huruf i kecil = (i), sedangkan Investment/investasi disimbolkan dengan huruf I besar = (I).       
    """)
    st.write("4. Profit (Laba)")
    st.write("""Profit/Laba adalah pendapatan yang diterima oleh pemilik Faktor Produksi Keahlian / Kewirausahaan (Entrepreneurship). Wirausahawan adalah orang yang menanggung risiko, menggabungkan tanah, tenaga kerja, dan modal agar tercipta proses produksi yang sukses.
    Laba dengan kata lain adalah sisa hasil usaha setelah semua biaya (upah, sewa, bunga) dibayarkan.
    """)
    st.write("""Di soal-soal, komponen Laba biasanya dipecah atau diubah menjadi Laba ditahan (Keuntungan yang tidak dibagikan, melainkan disimpan perusahaan untuk modal lagi),
    Dividen (Bagian keuntungan yang dibagikan kepada pemegang saham), dan Pajak Penghasilan Perusahaan (Bagian laba yang disetor ke negara).
    Jika di soal ketiga komponen di atas muncul terpisah, kita harus menjumlahkan ketiganya terlebih dahulu untuk mendapatkan total nilai Laba.
    """)
    st.write("Bagaimana, mudah bukan? Jika masih belum dipahami bisa discroll ke atas dan dipahami lagi, ya.")

    st.info("Jadi, Rumus menghitung pendekatan pendapatan yaitu Y = r + w + i + p")

    st.divider()
    st.header("Yuk, kita latihan soal!")
    st.write("Kerjakan soal di bawah ini dengan teliti, ya!")

    st.markdown("""
    Hitunglah besarnya Pendapatan Nasional menggunakan Pendekatan Pendapatan berdasarkan data berikut!

    Data Keuangan Negara Maju Jaya (dalam miliar rupiah)

    - Upah tenaga kerja = Rp450.000
    - Pengeluaran Konsumsi Rumah Tangga = Rp500.000
    - Pendapatan Sewa Tanah Kantor = Rp120.000
    - Pengeluaran Investasi Perusahaan = Rp350.000
    - Pembayaran Dividen kepada Pemegang Saham = Rp80.000
    - Laba Ditahan Perusahaan = Rp50.000
    - Pendapatan Bunga Neto dari Modal = Rp60.000
    - Pajak Keuntungan Perusahaan = Rp30.000
    - Pengeluaran Pemerintah = Rp400.000
    - Ekspor Neto = Rp90.000
    - Tunjangan Hari Raya (THR) Pegawai = Rp50.000
    """)

    st.markdown("Berapakah Pendapatan Nasional negara tersebut?")

    jawaban_pn = st.radio(
        "Pilih jawaban yang benar:",
        (
            "A. Rp1.340.000 miliar",
            "B. Rp840.000 miliar",
            "C. Rp790.000 miliar",
            "D. Rp710.000 miliar",
            "E. Rp680.000 miliar"
        ),
        index=None,
        key="soal_pendapatan"
    )

    if st.button("Periksa Jawaban", key="btn_pendapatan"):

        if jawaban_pn == "B. Rp840.000 miliar":

            st.balloons()

            st.success("Jawaban kamu benar!")

            st.info("""
    Langkah Penyelesaian:

    Y = w + r + i + p

    w (upah & gaji):
    450.000 + 50.000 = 500.000

    r (sewa):
    120.000

    i (bunga):
    60.000

    p (profit):
    80.000 + 50.000 + 30.000 = 160.000

    Total:
    500.000 + 120.000 + 60.000 + 160.000
    = Rp840.000 miliar
    """)

        else:

            st.error("Jawaban kamu masih kurang tepat!")

            if jawaban_pn == "A. Rp1.340.000 miliar":
                st.warning("""
    Kamu kemungkinan memakai pendekatan pengeluaran 
    (C + I + G + X - M), padahal soal meminta pendekatan pendapatan.
    """)

            elif jawaban_pn == "C. Rp790.000 miliar":
                st.warning("""
    Kamu kemungkinan lupa memasukkan THR Pegawai 
    ke dalam komponen upah (w).
    """)

            elif jawaban_pn == "D. Rp710.000 miliar":
                st.warning("""
    Kamu kemungkinan hanya memasukkan dividen untuk profit (p), 
    dan lupa laba ditahan serta pajak keuntungan.
    """)

            elif jawaban_pn == "E. Rp680.000 miliar":
                st.warning("""
    Kamu kemungkinan tidak memasukkan komponen profit perusahaan (p).
    """)

    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"
   

# PENDEKATAN PENGELUARAN
elif st.session_state.active_tab == "tab5":
    st.header("Konsep Pendekatan Pengeluaran")
    st.info("""
    Kamu bisa menggunakan menu ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Pengeluaran. Selamat belajar, ya!
    """)

    st.write("""Sebelum kita memulai perhitungan, alangkah baiknya kita mengenal dahulu komponen-komponen apa saja yang masuk ke dalam perhitungan
    pendekatan pengeluaran. Seperti yang kita ketahui, rumus menghitungnya yaitu Y = C + G + I + (X - M).
    Inti dari pendekatan pengeluaran (expenditure approach) sebenarnya adalah menghitung total nilai belanja atau pengeluaran yang dilakukan oleh seluruh
    pelaku ekonomi (RTK, Perusahaan, Pemerintah, dan Masyarakat Luar Negeri) untuk membeli barang dan jasa akhir yang diproduksi di dalam negeri selama satu tahun.
    """)
    st.info("Mengapa ya menghitung pengeluaran bisa menghasilkan angka Pendapatan Nasional?")
    st.write("""Jadi seperti ini, setiap uang yang kita keluarkan untuk belanja, otomatis akan menjadi pendapatan bagi orang lain yang menjual barang tersebut.
    Jika kita mendata dan menjumlahkan semua nota belanjaan dari seluruh pelosok negeri, maka total nilai belanjaan itu nilainya akan sama persis
    dengan total pendapatan yang diterima oleh seluruh masyarakat di negara tersebut.""")

    st.write("So, Pengeluaran = Pendapatan. Hanya pindah tangan saja, dari pembeli ke penjual.")
    st.info("Ingat aturan emas: Pendekatan Pengeluaran melihat perekonomian dari sudut pandang konsumen/pembeli")
    st.divider()
    st.subheader("Yuk, kita pahami dulu semua komponennya!")
    st.write("1. Konsumsi (Consumption) = C")
    st.markdown("""Komponen konsumsi (C) menghitung seluruh pengeluaran yang dilakukan oleh individu atau rumah tangga konsumen untuk membeli barang dan jasa demi memenuhi kebutuhan sehari-hari.
    Terdapat tiga jenis barang belanjaan disini, yaitu:
             - Barang Tahan Lama (Durable Goods), yaitu barang yang habisnya lama (lebih dari 3 tahun). Contohnya: Mobil pribadi, laptop kuliah, kulkas, kasur.
             - Barang Tidak Tahan Lama (Nondurable Goods), yaitu barang yang langsung habis atau memiliki masa pakai pendek. Contohnya: Makanan, bensin, sabun, baju.
             - Biaya Jasa (Services) yaitu pembayaran atas layanan. Contohnya: Biaya potong rambut, tiket bioskop, paket data internet, jasa ojek online.
    """)
    st.info("Eits, pembelian Rumah Tinggal Baru oleh perorangan TIDAK masuk ke komponen C ya, melainkan dimasukkan ke komponen Investasi (I) karena properti dianggap sebagai aset produktif jangka panjang. Selain itu, laptop yang dibeli perusahaan untuk operasional karyawannya juga termasuk ke dalam Investment (I).")
    
    st.write("2. Investasi (Investment) = I")
    st.write("""Dalam makroekonomi, Investasi (I) bukanlah uang yang ditanam di saham atau kripto. Investasi di sini adalah pengeluaran untuk membeli barang modal fisik yang digunakan untuk menghasilkan barang/jasa lain di masa depan.
    Yang belanja di komponen ini utamanya adalah sektor dunia usaha yaitu perusahaan.
    """)
    st.markdown("""Kategori belanjanya antara lain, yaitu:
             - Investasi fisik contohnya membangun pabrik baru, mendirikan gedung kantor, atau membeli ruko.
             - Membeli peralatan modal seperti mesin cetak, truk angkutan barang, komputer untuk operasional karyawan.
             - Stok persediaan barang yang sudah diproduksi tahun ini tapi belum laku dijual dan masih tersimpan di gudang tetap dihitung sebagai pengeluaran perusahaan "membeli" produknya sendiri agar tetap tercatat di PDB tahun berjalan.
             - Rumah tinggal baru yang dibeli perorangan.
    """)
    st.info("Eits, Investasi Keuangan seperti membeli saham, obligasi, atau reksa dana SAMA SEKALI TIDAK DIHITUNG karena hanya berupa perpindahan kepemilikan kertas/aset, dan tidak menciptakan barang fisik baru di dalam perekonomian. Komponen yang dicatat murni berupa investasi fisik, seperti membangun pabrik, membeli mesin, atau menambah stok persediaan barang.")
    
    st.write("3. Pengeluaran Pemerintah (Government Purchases) = G")
    st.write("""Komponen ini menghitung semua pengeluaran yang dilakukan oleh pemerintah (baik pusat maupun daerah) untuk membeli barang, jasa, maupun menggaji instansi negara demi menjalankan roda pemerintahan.
    Komponen G hanya menghitung pengeluaran pemerintah yang bersifat produktif (ada barang atau jasa yang diterima negara sebagai timbal balik), seperti membayar gaji PNS, membangun jalan tol, atau membeli alutsista.
    """)
    
    st.info("Eits, pengeluaran pemerintah yang sifatnya sepihak seperti Bantuan Langsung Tunai (BLT), subsidi BBM, atau beasiswa pendidikan disebut Transfer Payment (pembayaran pindahan) dan TIDAK boleh dimasukkan ke dalam rumus pendapatan nasional karena tidak menghasilkan barang/jasa baru saat itu juga.")
    
    st.write("4. Ekspor (Exports) = X")
    st.write("""Ekspor menghitung pengeluaran yang dilakukan oleh orang, perusahaan, atau pemerintah luar negeri untuk membeli barang dan jasa yang diproduksi di dalam negeri kita.
    """)
    st.write("Ketika kita melihat rumus, mengapa X bernilai plus/ditambah ya? Yap, itu karena barangnya diproduksi di dalam negeri menggunakan tenaga kerja dan sumber daya kita, maka pengeluaran orang asing tersebut harus menambah nilai Pendapatan Nasional kita. Contohnya turis asing yang menyewa hotel di Bali termasuk ekspor jasa, atau pengusaha Jepang membeli batu bara dari Kalimantan termasuk ekspor barang.")

    st.write("5. Impor (Imports) = M")
    st.write("""Komponen Impor menghitung pengeluaran masyarakat atau negara kita untuk membeli barang dan jasa yang diproduksi oleh negara lain.
    """)
    st.write("""Mengapa di rumus komponen M bernilai minus ya? Impor diberi tanda minus bukan karena uang kita keluar, melainkan sebagai faktor koreksi.
    Contoh kasusnya seperti ini, ketika kita membeli HP buatan Amerika seharga Rp15 juta, transaksi itu otomatis tercatat di komponen Konsumsi (C) kit sebesar Rp15 juta. Padahal, HP itu tidak diproduksi di Indonesia. Agar angka PDB kita tidak membengkak oleh barang buatan luar negeri, maka di bagian akhir rumus, nilai Rp15 juta tadi harus dikurangkan melalui komponen Impor (M).Jadi, fungsi minus pada M adalah untuk menghapus pengeluaran barang luar negeri yang telanjur tercatat di komponen C, I, atau G.
    """)
    st.info("""Eits, ada logika terbalik nih antara ekspor dan impor. Pasti kita sering berpikir: Lho, 'kan kalau kita impor barang, kita mengeluarkan uang untuk belanja. Kenapa di pendekatan pengeluaran malah dikurangi?
    Impor diberi tanda minus karena barang impor tersebut sudah telanjur ikut terhitung di komponen Konsumsi (C) atau Investasi (I) masyarakat sebelumnya. Jadi, tanda minus pada Impor (M) berfungsi sebagai faktor pengurang agar kita benar-benar hanya menghitung pengeluaran untuk barang yang diproduksi di dalam negeri saja. Menjelaskan logika pelurusan ini ke siswa SMA biasanya membutuhkan analogi yang kuat.
    """)
    st.write("Bagaimana, apakah sudah paham? Yuk scroll ke atas dan pahami kembali jika masih kesulitan.")
    st.divider()
    st.header("Yuk, kita latihan soal!")
    st.subheader("Kerjakan soal berikut dengan teliti, ya.")

    st.markdown("""
    Diketahui data ekonomi makro Negara X sebagai berikut (dalam triliun rupiah):

    - Konsumsi rumah tangga = Rp400
    - Investasi pembelian saham di bursa efek = Rp80
    - Pengeluaran pemerintah untuk gaji PNS = Rp150
    - Ekspor barang ke Jepang = Rp90
    - Keuntungan yang diperoleh perusahaan = Rp120
    - Pembelian mesin pabrik baru oleh pengusaha = Rp110
    - Pembayaran bantuan sosial (bansos) oleh pemerintah = Rp40
    - Impor kosmetik dari Korea = Rp50

    Berapakah besarnya Pendapatan Nasional Negara X dengan pendekatan pengeluaran?
    """)

    jawaban_pengeluaran = st.radio(
        "Pilih jawaban yang benar:",
        (
            "A. Rp840 triliun",
            "B. Rp780 triliun",
            "C. Rp740 triliun",
            "D. Rp700 triliun"
        ),
        index=None,
        key="soal_pengeluaran"
    )

    if st.button("Periksa Jawaban", key="btn_pengeluaran"):

        if jawaban_pengeluaran == "D. Rp700 triliun":

            st.balloons()

            st.success("Jawaban kamu benar!")

            st.info("""
    
    Rumus Pendekatan Pengeluaran:

    Y = C + I + G + (X - M)

    Data yang digunakan:
    - C = 400
    - I = 110 (mesin pabrik)
    - G = 150 (gaji PNS)
    - X = 90
    - M = 50

    Perhitungan:
    Y = 400 + 110 + 150 + (90 - 50)

    Y = 660 + 40

    Y = Rp700 triliun
    """)

        else:

            st.error("Jawaban kamu masih kurang tepat!")

            if jawaban_pengeluaran == "A. Rp840 triliun":
                st.warning("""Kamu kemungkinan memasukkan semua data tanpa menyaring mana yang termasuk komponen pendekatan pengeluaran.
    """)

            elif jawaban_pengeluaran == "B. Rp780 triliun":
                st.warning("""Kamu kemungkinan masih memasukkan investasi saham atau keuntungan perusahaan ke dalam perhitungan.
    """)

            elif jawaban_pengeluaran == "C. Rp740 triliun":
                st.warning("""Kamu kemungkinan memasukkan bansos pemerintah sebagai komponen G, padahal bansos termasuk transfer payment.
    """)
    
    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# PENDAPATAN PERKAPITA
elif st.session_state.active_tab == "tab6":
    def pendapatan_perkapita():
        st.title("Pendapatan Perkapita")
    
    st.header("Konsep Pendapatan Perkapita")
    st.subheader("1. Hakikat Pendapatan Perkapita")
    st.markdown("""Secara harfiah, per capita berasal dari bahasa Latin yang berarti "per kepala". 
    Jadi, Pendapatan Perkapita adalah ukuran rata-rata pendapatan yang diperoleh oleh setiap penduduk di suatu negara dalam kurun waktu satu tahun.""")
    
    st.info("""Pendapatan perkapita digunakan oleh lembaga internasional seperti Bank Dunia sebagai indikator utama untuk mengukur tingkat kemakmuran suatu negara dan mengelompokkan negara tersebut
    apakah masuk kategori Low, Lower-Middle, Upper-Middle, atau High Income.
    """)

    st.divider()
    st.subheader("2. Rumus dan Komponen Penting")
    st.image('rumuspp1.jpg')
    st.write("""Rumus perhitungannya memang terlihat sederhana, namun kita sering bingung yang dimasukkan nilai PDB atau nilai PNB, ya?
    Ingat, PDB/GDP adalah rata-rata output yang dihasilkan di dalam wilayah domestik suatu negara, tanpa memedulikan apakah itu milik warga lokal atau perusahaan asing.
    Sedangkan, PNB/GNP adalah rata-rata pendapatan yang benar-benar dimiliki oleh Warga Negara tersebut (baik yang tinggal di dalam negeri maupun di luar negeri).
    Pada pendapatan perkapita yang kita hitung adalah tingkat kesejahteraan warga negara kita yang sebenarnya, sehinga perhitungan menggunakan PNB/GNP jauh lebih akurat daripada PDB/GDP.
    """)
    st.divider()
    st.subheader("3. GDP Deflator dan Riil")
    st.write("""Sebelum melangkah lebih jauh, kita harus memahami apa itu GDP Deflator dan Riil.
    """)
    st.write("A. GDP Deflator")
    st.write("""
    GDP Deflator adalah rasio atau perbandingan antara apa yang diproduksi saat ini menggunakan harga saat ini (Nominal) dengan apa yang diproduksi saat ini menggunakan harga tahun dasar (Riil).
    Fungsi GDP Deflator adalah untuk menyaring data kotor. Misalnya pemerintah punya data kotor yang masih terkena inflasi, yang disebut GDP Nominal. Pemerintah ingin mengubahnya menjadi data bersih, yaitu GDP Riil. Alat saringnya dinamakan GDP Deflator.
    Rumus GDP Deflator, yaitu:
    """)
    st.image('deflator.png')
    st.info("""
    Patokan angka dari GDP Deflator pada tahun dasar adalah 100. Jadi, hasil akhir dari rumus di atas akan menunjukkan seberapa besar inflasi yang terjadi.
    Jika hasilnya 100 artinya tidak ada perubahan harga atau tidak ada inflasi yang terjadi dibanding tahun dasar. Jika hasilnya 125 artinya telah terjadi kenaikan harga atau inflasi sebesar 25% (125-100) sejak tahun dasar.
    Jika hasilnya 108 artinya terjadi inflasi sebesar 8% (108-100).
    """)
    st.write("Coba perhatikan contoh soal di bawah ini:")
    st.write("""
    Negara Asgard pada tahun 2025 memiliki data GDP Nominal sebesar Rp5.500 triliun, sedangkan data GDP Riilnya adalah Rp5.000 triliun. Hitunglah besarnya GDP Deflator negara tersebut!
    """)
    st.markdown("""
    Penyelesaian:       
    - GDP Deflator = Rp5.500/Rp5.000 x 100
    - GDP Deflator = 1,1 x 100
    - GDP Deflator = 110
    """)
    st.write("Mudah dipahami bukan?")

    st.write("B. GDP Riil")
    st.write("GDP Riil adalah data ekonomi yang sudah bersih karena efek inflasi sudah disaring menggunakan GDP Deflator tadi. Rumusnya, yaitu:")
    st.image('gdpriil.png')
    st.write("Coba kita cek soal di bawah ini:")
    st.markdown("""
    Pada tahun 2025, Negara Fufufafa mencatat data makroekonomi sebagai berikut:
    - Gross Domestic Product (GDP) Nominal = Rp120.000.000
    - Indeks Harga (GDP Deflator) = 120
    Berdasarkan data tersebut, hitunglah besarnya GDP Riil Negara Fufufafa!
    """)
    st.markdown("""
    Penyelesaian:
    - GDP Riil = Rp120.000.000/120 x 100
    - GDP Riil = Rp1.000.000 x 100
    - GDP Riil = Rp10.000.000
    Nilai GDP Riil negara Fufufafa sebesar Rp10.000.000
    """)
    st.divider()
    st.subheader("4. Jenis Pendapatan Perkapita")
    st.write("Terdapat dua jenis pendapatan perkapita, antara lain:")
    st.write("1. Pendapatan Perkapita Atas Dasar Harga Berlaku (Nominal).")
    st.markdown("""Pendapatan perkapita atas dasar harga berlaku adalah nilai pendapatan rata-rata penduduk yang dihitung menggunakan harga barang dan jasa yang sedang berlaku pada tahun berjalan.
    Jika kita ingin menghitung pendapatan perkapita tahun 2026, maka seluruh output ekonomi negara di tahun 2026 akan dikalikan dengan harga pasar di tahun 2026 pula, lalu dibagi dengan jumlah penduduk pada tahun tersebut.""")
    st.markdown("""
    - Angka ini mencerminkan kondisi nilai mata uang secara aktual pada tahun tersebut.
    - Angka nominal ini dapat bersifat menipu. Jika pendapatan perkapita nominal suatu negara melonjak dari tahun lalu, hal itu belum tentu menandakan bahwa masyarakatnya menjadi lebih makmur. Peningkatan tersebut bisa saja terjadi bukan
    karena jumlah produksi barang bertambah, melainkan karena harga-harga barang di pasar sedang mengalami kenaikan yang masif (Inflasi).
    """)
    st.write("Rumus menghitung pendapatan perkapita nominal")
    st.image('rumusppnom.png')

    st.write("2. Pendapatan Perkapita Atas Dasar Harga Konstan (Riil)")
    st.write("""Pendapatan perkapita atas dasar harga konstan adalah nilai pendapatan rata-rata penduduk yang dihitung
    menggunakan patokan harga pada tahun dasar tertentu yang dinilai stabil (misalnya, menggunakan patokan harga tahun 2010 atau tahun acuan lainnya).
    Jadi, meskipun kita menghitung pendapatan perkapita untuk tahun 2024, 2025, atau 2026, harga barang yang digunakan sebagai komponen pengali tetap dikunci pada harga tahun dasar.
    """)
    st.markdown("""
    - Angka ini sudah dibersihkan atau dieliminasi dari dampak inflasi.
    - Pendapatan perkapita riil inilah yang menjadi indikator akurat untuk mengukur kemakmuran yang sesungguhnya. Jika pendapatan perkapita harga konstan suatu negara meningkat,
    dapat dipastikan bahwa volume produksi barang dan jasa di negara tersebut memang benar-benar bertambah secara fisik, yang berarti daya beli riil masyarakatnya mengalami perbaikan.
    """)
    st.write("Rumus menghitung pendapatan perkapita riil: ")
    st.image('rumusppriil.png')
    st.write("Supaya semakin paham, perhatikan contoh soal berikut:")
    st.markdown("""
    Misalkan di sebuah negara kecil bernama Midgard, datanya seperti ini:
    - GDP Nominal = Rp1.200.000
    - GDP Deflator = 120
    - Jumlah Penduduk = 10 orang.
    Berdasarkan data di atas, hitunglah pendapatan perkapita nominal dan riil-nya!
    """)
    st.write("Cara menghitungnya jangan langsung dibagi dengan jumlah penduduk, melainkan kita saring dulu dengan alat GDP Deflator menggunakan rumus GDP Riil.")
    st.write("""
    GDP Riil = Rp1.200.000/120 x 100 = Rp1.000.000. Jadi, GDP Riil-nya sebesar Rp1.000.000, nilai ini sudah bebas dari inflasi.
    """)
    st.write("Kita sudah mempunyai satu data tambahan nih, yaitu GDP Riil = Rp1.000.000. Selanjutnya tingga kita masukkan seja ke rumus pendapatan perkapita nominal dan riil.")
    st.write("a. Pendapatan perkapita nominal")
    st.write("Pendapatan perkapita nominal = Rp1.200.000/10 = Rp120.000/orang")
    st.write("b. Pendapatan perkapita riil")
    st.write("Pendapatan perkapita riil = Rp1.000.000/10 = Rp100.000/orang")
    st.warning("Psst, kalau masih ada kesulitan bisa dibaca dan dipahami kembali, ya.")
    st.divider()
    st.write("Berikut ringkasan perbandingan pendapataan perkapita nominal dan riil:")
    st.image('ssppnppr.png')
    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# DISTRIBUSI PENDAPATAN
elif st.session_state.active_tab == "tab7":
    def distribusi_pendapatan():
     st.title("Distribusi Pendapatan")

    st.header("Konsep Distribusi Pendapatan")
    st.subheader("1. Pengertian Distribusi Pendapatan")
    st.markdown("""Distribusi pendapatan adalah cerminan penyebaran pendapatan yang dihasilkan suatu negara kepada seluruh lapisan masyarakatnya.
    Dalam ekonomi, kita tidak hanya mengejar Pertumbuhan Ekonomi, tetapi juga Pemerataan Pembangunan yaitu memperhatikan kesejahteraan masyarakat, salah satunya
    dengan melihat apakah pendapatan negara kita merata tidak. Jika Pendapatan Nasional (PDB)
    tinggi tapi hanya dinikmati oleh segelintir orang, maka terjadi ketimpangan ekonomi yang parah.""")
    
    st.markdown("""Ada dua cara pandang dalam melihat distribusi ini:
    - Distribusi Ukuran (Size Distribution) yaitu langsung melihat berapa besar pendapatan yang diterima oleh perorangan atau rumah tangga, tanpa peduli dari mana sumber pendapatan itu (apakah dari gaji, warisan, atau sewa).
    - Distribusi Fungsional (Functional Distribution) yaitu melihat pembagian pendapatan berdasarkan faktor produksi yang disediakan (buruh mendapat upah, pemilik modal mendapat untung, pemilik tanah mendapat sewa).
    """)
    st.divider()
    st.subheader("2. Pengertian Kurva Lorenz")
    st.write("Perhatikan gambar kurva lorenz berikut:")
    st.image('kurva_lorenz.png')
    st.write("Kurva Lorenz adalah grafik yang digunakan untuk menunjukkan perbandingan antara persentase kumulatif penduduk dan persentase kumulatif pendapatan yang mereka terima.")
    st.markdown("""
    - Sumbu Horizontal (X) menunjukkan persentase kumulatif penduduk (dari 0% sampai 100%).
    - Sumbu Vertikal (Y) menunjukkan persentase kumulatif pendapatan (dari 0% sampai 100%).
    - Garis Kemerataan Sempurna (Garis Diagonal 45 derajat) yaitu garis lurus yang membelah grafik secara diagonal. Garis ini adalah kondisi ideal hipotetis. 
        Artinya, 20% penduduk mendapat 20% pendapatan, 50% penduduk mendapat 50% pendapatan, dan seterusnya.
    - Kurva Lorenz (Garis Melengkung) menunjukkan realitas distribusi pendapatan di suatu negara. Kurva ini selalu berada di bawah garis diagonal.
    """)
    st.info("Cara Membaca Kurva Lorenz: Semakin melengkung ke bawah (menjauhi garis diagonal) Kurva Lorenz tersebut, artinya distribusi pendapatan di negara itu semakin timpang. Sebaliknya, semakin merapat kurva ke garis diagonal, artinya pendapatan semakin merata. Jangan sampai terbalik, ya.")


    def gini_coefficient(income):
        sorted_income = np.sort(income)
        n = len(income)
        cum_income = np.cumsum(sorted_income)
        total_income = cum_income[-1]
        
        gini = (2 / n) * np.sum((np.arange(1, n + 1) * sorted_income)) / total_income - (n + 1) / n
        return gini, sorted_income, cum_income

    def plot_lorenz_curve(sorted_income, cum_income):
        lorenz_x = np.arange(1, len(sorted_income) + 1) / len(sorted_income)
        lorenz_y = cum_income / cum_income[-1]
        
        plt.figure(figsize=(8, 6))
        plt.plot(lorenz_x, lorenz_y, label="Kurva Lorenz", color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Garis Kesetaraan")
        plt.title("Kurva Lorenz")
        plt.xlabel("Persentase Penduduk (%)")
        plt.ylabel("Persentase Pendapatan (%)")
        plt.legend()
        plt.grid(True)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        return img_buf

    st.write("Yuk, coba kita simulasikan bagaimana kurva lorenz bekerja.")
    st.write("Masukkan data pendapatan individu (pisahkan dengan koma, misalnya: 200, 500, 800, 1000)")
    income_input = st.text_input("Pendapatan (misalnya: 200, 500, 800, 1000)")

    if st.button("Kirim"):
        if income_input:
            try:
                income = [float(i) for i in income_input.split(",")]
                
                if len(income) < 2:
                    st.error("Masukkan lebih dari satu nilai pendapatan.")
                else:
                    gini, sorted_income, cum_income = gini_coefficient(income)
                    
                    st.success(f"Koefisien Gini: {gini:.4f}")
                    
                    img_buf = plot_lorenz_curve(sorted_income, cum_income)
                    st.image(img_buf, caption="Kurva Lorenz", use_container_width=True)

            except ValueError:
                st.error("Tolong masukkan angka yang valid.")
        else:
            st.error("Masukkan data pendapatan terlebih dahulu.")
    st.divider()
    st.subheader("3. Pengertian Koefisien Gini")
    st.write("""Koefisien Gini adalah indikator numerik yang mengukur derajat ketimpangan distribusi pendapatan secara keseluruhan.
    Nilainya diturunkan langsung dari Kurva Lorenz.
    """)
    st.write("Rumus menghitung koefisien gini:")
    st.image('rumusgini.png')
    st.markdown("""
    Jika kita melihat grafik Kurva Lorenz di atas, kita misalkan saja...
    - Area A adalah daerah yang berada di antara Garis Kemerataan Sempurna dan Kurva Lorenz.
    - Area B adalah daerah di bawah Kurva Lorenz.
    
    Karena total luas Segitiga di bawah garis diagonal (Area A + Area B) selalu bernilai tetap, maka...
    - Jika Kurva Lorenz berimpit dengan garis diagonal (merata sempurna), maka Luas Area A = 0. Berarti G = 0.
    - Jika Kurva Lorenz melengkung ekstrem hingga menyentuh pojok kanan bawah, maka Luas Area B = 0. Berarti G = 1.
    """)

    st.markdown("""
   Koefisien Gini memiliki kriteria ketimpangan sebagai berikut:
    - 0 = Distribusi merata sempurna
    - 0 - 0,4 = Ketimpangan rendah
    - 0,4 - 0,5 = Ketimpangan sedang
    - 0,5 - 1 = Ketimpangan tinggi
    - 1 = Distribusi tidak merata sempurna 
    """)

    st.warning("Semakin dekat ke 1 artinya ketimpangan semakin parah, dan semakin dekat ke 0 artinya semakin merata sempurna. Jangan sampai terbalik, ya")
    st.divider()
    st.subheader("4. Kriteria Bank Dunia")
    st.write("Selain Rasio Gini, Bank Dunia membuat klasifikasi alternatif yang lebih praktis untuk mengukur ketimpangan dengan cara membagi populasi penduduk menjadi 3 kelompok berdasarkan tingkat kesejahteraannya")
    st.markdown("""
    - Suatu negara yang kelompok 40% penduduk termiskinnya memperoleh pendapatan lebih kecil dari 12%,
    maka negara tersebut berada di tingkat ketimpangan yang tinggi dalam distribusi pendapatan.
    - Suatu negara yang kelompok 40% penduduk termiskinnya pendapatannya diantara 12% - 17 %, 
    maka negara tersebut berada di tingkat ketimpangan sedang dalam distribusi pendapatan.
    - Suatu negara yang kelompok 40 penduduk termiskinnya pendapatannya lebih dari 17%, 
    maka negara tersebut berada di tingkat ketimpangan rendah dalam distribusi pendapatan.
    """)

    st.markdown("""Misalkan Negara X memiliki Pendapatan Nasional Rp 1.000 Triliun.
    - Berdasarkan data, ternyata kelompok 40% penduduk paling miskin di negara tersebut setelah dijumlahkan pendapatannya hanya mengantongi total Rp 100 Triliun.
    - Artinya, mereka hanya menikmati 10% dari total pendapatan nasional.
    - Karena 10% itu kurang dari 12%, maka menurut Kriteria Bank Dunia, Negara X berada dalam kondisi Ketimpangan Tinggi.

    """)

    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# LATIHAN SOAL
elif st.session_state.active_tab == "tab8":
    questions = {
    "C1": [
        {
            "id": "C101",
            "question": "Total pendapatan yang diperoleh oleh seluruh pelaku ekonomi dalam suatu negara selama periode tertentu (biasanya satu tahun) disebut...",
            "options": [
                "A. Pendapatan per kapita",
                "B. Pendapatan nasional",
                "C. Produk domestik bruto",
                "D. Distribusi pendapatan",
                "E. Laba ditahan"

            ],
            "answer": "B. Pendapatan nasional",
            "explanation": "Secara definisi, total seluruh pendapatan yang diterima oleh seluruh pelaku ekonomi di suatu negara dalam periode tertentu (biasanya satu tahun) disebut Pendapatan Nasional."
            
        },
        {
            "id": "C102",
            "question":"Secara teori ekonomi, produk nasional bersih atau Net National Product (NNP) diperoleh dengan cara... ",
            "options":[
                "A. GDP + Pendapatan Neto Luar Negeri",
                "B. GNP dikurangi Pajak Tidak Langsung",
                "C. GNP dikurangi Penyusutan barang modal",
                "D. NNI ditambah Subsidi pemerintah",
                "E. PI dikurangi Pajak Langsung"

            ],
            "answer": "C. GNP dikurangi Penyusutan barang modal",
            "explanation": "Net National Product (NNP) atau Produk Nasional Neto adalah nilai produk nasional bersih. Karena selama proses produksi barang-barang modal (seperti mesin, alat, gedung) pasti mengalami aus atau penurunan nilai ekonomi, maka nilai GNP harus dibersihkan dengan cara dikurangi dengan penyusutan."
        },
        {
            "id": "C103",
            "question": "Di bawah ini yang merupakan komponen penambah (bersifat plus) dalam mencari nilai Personal Income (PI) dari NNI adalah... ",
            "options":[
                "A. Laba ditahan",
                "B. Pajak perseroan",
                "C. Iuran asuransi",
                "D. Transfer Payment",
                "E. Iuran jaminan sosial"

            ],
            "answer": "D. Transfer Payment",
            "explanation": "Transfer Payment (tunjangan sosial, beasiswa, pensiunan) bersifat menambah (plus) karena merupakan aliran uang dari pemerintah langsung ke masyarakat tanpa masyarakat harus bekerja/menyerahkan faktor produksi pada tahun tersebut."
        },
        {
            "id": "C104",
            "question": "Pendapatan pribadi yang sudah bersih dari segala kewajiban hukum terhadap negara dan sepenuhnya siap digunakan untuk konsumsi atau tabungan disebut... ",
            "options": [
                "A. Gross Domestic Product", 
                "B. Net National Product",
                "C. Personal Income",
                "D. Disposable Income", 
                "E. Net National Income"

            ],
            "answer": "D. Disposable Income",
            "explanation": "Disposable Income (DI) atau Pendapatan yang Siap Dibelanjakan adalah sisa pendapatan pribadi yang telah dikurangi dengan Pajak Langsung (seperti Pajak Penghasilan/PPh). Pendapatan inilah yang sudah 100% bersih dan siap digunakan secara bebas oleh pemiliknya, baik untuk keperluan konsumsi (C) maupun untuk ditabung (S)."
        },
        {
        "id": "C105",
        "question": "Rumus matematika untuk menghitung pendapatan nasional dengan pendekatan pendapatan adalah Y = r + w + i + p. Komponen huruf w dan p dalam rumus tersebut melambangkan...",
        "options": [
            "A. Rent (Sewa) dan Interest (Bunga)",
            "B. Wage (Upah/Gaji) dan Profit (Laba Usaha)",
            "C. West (Barat) dan Price (Harga)",
            "D. Wealth (Kekayaan) dan Population (Penduduk)",
            "E. Warranty (Garansi) dan Product (Produk)"
        ],
        "answer": "B. Wage (Upah/Gaji) dan Profit (Laba Usaha)",
        "explanation": "Dalam pendekatan pendapatan, Y = r + w + i + p, di mana r = rent, w = wage, i = interest, dan p = profit."
    },

    {
        "id": "C106",
        "question": "Ukuran rata-rata pendapatan yang diperoleh oleh setiap penduduk di suatu negara dalam kurun waktu satu tahun disebut...",
        "options": [
            "A. Produk Domestik Bruto regional",
            "B. Pendapatan Perkapita",
            "C. Koefisien Gini Nasional",
            "D. Nilai Tambah Bruto",
            "E. Kurva Kesetaraan Ekonomi"
        ],
        "answer": "B. Pendapatan Perkapita",
        "explanation": "Pendapatan per kapita diperoleh dari pembagian pendapatan nasional dengan jumlah penduduk."
    }
    ],

    "C2": [
        {
            "id": "C201",
            "question": "Suatu negara mengalami peningkatan kualitas pendidikan dan keterampilan tenaga kerja. Dampak yang paling mungkin terjadi terhadap pendapatan nasional adalah ... ",
            "options": [
                "A. Pendapatan nasional menurun karena biaya pendidikan meningkat",
                "B. Pendapatan nasional tetap karena pendidikan tidak berhubungan dengan produksi",
                "C. Pendapatan nasional meningkat karena produktivitas tenaga kerja bertambah",
                "D. Pendapatan nasional menurun karena jumlah penduduk berkurang",
                "E. Pendapatan nasional hanya memengaruhi sektor jasa"
            ],
            "answer": "C. Pendapatan nasional meningkat karena produktivitas tenaga kerja bertambah",
            "explanation": "Kualitas sumber daya manusia yang lebih baik akan meningkatkan produktivitas tenaga kerja sehingga jumlah barang dan jasa yang dihasilkan bertambah. Akibatnya, pendapatan nasional cenderung meningkat."

        },
        {
            "id": "C202",
            "question": "Konsep Gross Domestic Product (GDP) menganut asas teritorial (wilayah). Arti dari asas teritorial tersebut adalah... ",
            "options": [
                "A. Hanya menghitung pendapatan warga negara lokal saja",
                "B. Faktor produksi milik siapapun (WNI maupun WNA) yang beroperasi di dalam negeri, nilainya mutlak dihitung",
                "C. Hanya menghitung pendapatan yang berasal dari luar negeri",
                "D. Menghitung pendapatan warga lokal yang berada di luar negeri saja",
                "E. Menghapus semua pajak dari wilayah domestik",

            ],
            "answer": "B. Faktor produksi milik siapapun (WNI maupun WNA) yang beroperasi di dalam negeri, nilainya mutlak dihitung",
            "explanation":"Gross Domestic Product (GDP) atau Produk Domestik Bruto (PDB) menitikberatkan pada aspek domestik/wilayah/teritorial. Artinya, semua aktivitas produksi yang menghasilkan barang dan jasa di dalam batas wilayah geografi suatu negara baik dilakukan oleh warga negara asli (WNI) maupun warga negara asing (WNA) akan dihitung ke dalam GDP negara tersebut."

        },
        {
            "id": "C203",
            "question": "Konsep Gross National Product (GNP) menganut asas nasionalitas atau kewarganegaraan. Berdasarkan asas ini, untuk mendapatkan nilai GNP maka... ",
            "options": [
                "A. GDP harus dikurangi penyusutan mesin pabrik",
                "B. GDP disesuaikan dengan Pendapatan Neto terhadap Luar Negeri",
                "C. GDP langsung dikurangi pajak langsung",
                "D. GDP ditambah dengan nilai laba ditahan",
                "E. GDP dibagi dengan jumlah penduduk"

            ],
            "answer": "B. GDP disesuaikan dengan Pendapatan Neto terhadap Luar Negeri",
            "explanation": "Gross National Product (GNP) atau Produk Nasional Bruto (PNB) menitikberatkan pada aspek nasionalitas/kewarganegaraan."
        },
        {
            "id": "C204",
            "question":"Dalam menghitung Net National Income (NNI), mengapa Pajak Tidak Langsung harus dikurangkan dan Subsidi harus ditambahkan ke nilai NNP? ",
            "options":[
                "A. Karena subsidi bersifat merugikan kas perusahaan",
                "B. Pajak tidak langsung adalah titipan pemerintah yang menempel pada produk (bukan omset murni), sedangkan subsidi meringankan modal riil hasil usaha",
                "C. Karena pajak tidak langsung selalu dibayarkan oleh wajib pajak sendiri dan tidak bisa dialihkan",
                "D. Agar nilai NNI sama persis dengan nilai inflasi pasar",
                "E. Karena subsidi merupakan bentuk tabungan jangka panjang masyarakat",

            ],
            "answer": "B. Pajak tidak langsung adalah titipan pemerintah yang menempel pada produk (bukan omset murni), sedangkan subsidi meringankan modal riil hasil usaha",
            "explanation":"Pengurangan pajak tidak langsung dan penambahan subsidi dilakukan agar nilai NNI mencerminkan pendapatan bersih yang benar-benar diterima oleh pemilik faktor produksi dalam suatu perekonomian."
        },
        {
            "id": "C205",
            "question":"""
            Perhatikan beberapa kegiatan ekonomi berikut!
            - 1. Sebuah pabrik sepatu membeli kulit untuk diolah menjadi sepatu. 
            - 2. Seorang siswa membeli sepatu untuk digunakan ke sekolah. 
            - 3. Sebuah toko roti membeli tepung untuk membuat roti. 
            - 4. Seorang ibu membeli roti untuk dikonsumsi keluarganya. 
            Berdasarkan konsep barang antara dan barang akhir, pasangan yang termasuk barang akhir adalah ...
            """,
            "options": [
                "A. (1) dan (3)",
                "B. (1) dan (2)",
                "C. (2) dan (4)",
                "D. (2) dan (3)",
                "E. (3) dan (4)"

            ],
            "answer": "C. (2) dan (4)",
            "explanation": """
            Barang akhir (final goods) adalah barang yang dibeli untuk langsung digunakan atau dikonsumsi oleh konsumen akhir dan tidak diproses kembali untuk dijual.
            •	Kulit yang dibeli pabrik sepatu termasuk barang antara, karena akan diolah kembali.
            •	Sepatu yang dibeli siswa termasuk barang akhir, karena digunakan langsung oleh konsumen.
            •	Tepung yang dibeli toko roti termasuk barang antara, karena digunakan sebagai bahan produksi.
            •	Roti yang dibeli ibu rumah tangga termasuk barang akhir, karena langsung dikonsumsi.
            """
        },
        {
            "id":"C206",
            "question": "Di bawah ini contoh aktivitas pembelian yang tergolong ke dalam Barang Akhir (Final Goods) adalah... ",
            "options": [
                "A. Tepung dibeli pabrik roti untuk membuat kue",
                "B. Gula dibeli perusahaan minuman untuk produksi massal", 
                "C. Ibu rumah tangga membeli 1 kg telur di pasar untuk digoreng sebagai lauk sarapan",
                "D. Kain dibeli konveksi untuk dijahit menjadi kemeja",
                "E. Kapas dibeli pabrik pemintal untuk dijadikan benang"

            ],
            "answer": "C. Ibu rumah tangga membeli 1 kg telur di pasar untuk digoreng sebagai lauk sarapan",
            "explanation": "Telur yang dibeli oleh ibu rumah tangga langsung dikonsumsi habis oleh konsumen akhir dan tidak dijual atau diolah lagi demi keuntungan ekonomis berantai."
        },
        {
            "id": "C207",
            "question": "Mengapa nilai barang antara (intermediate goods) TIDAK boleh dimasukkan dalam perhitungan Pendapatan Nasional?",
            "options": [
                "A. Karena barang antara tidak memiliki harga jual di pasar",
                "B. Untuk menghindari kesalahan fatal berupa penghitungan ganda (double counting)",
                "C. Karena barang antara hanya diproduksi oleh pemerintah saja",
                "D. Untuk mempercepat proses pencatatan nota oleh akuntan negara",
                "E. Karena nilainya selalu berubah-ubah setiap hari"
            ],
            "answer": "B. Untuk menghindari kesalahan fatal berupa penghitungan ganda (double counting)",
            "explanation": "Nilai barang antara sudah terkandung dalam nilai barang akhir sehingga tidak boleh dihitung dua kali."
        },

        {
            "id": "C208",
            "question": "Inti dasar dari penghitungan pendapatan nasional menggunakan metode Pendekatan Pengeluaran (Expenditure Approach) adalah...",
            "options": [
                "A. Menjumlahkan nilai tambah dari semua sektor produksi berantai",
                "B. Menghitung total nilai belanja atau pengeluaran yang dilakukan oleh seluruh pelaku ekonomi dalam satu tahun",
                "C. Mengumpulkan data modal pinjaman dari luar negeri saja",
                "D. Menghitung balas jasa berupa upah dan sewa yang diterima RTK",
                "E. Menghitung sisa hasil usaha bersih milik dinas perpajakan"
            ],
            "answer": "B. Menghitung total nilai belanja atau pengeluaran yang dilakukan oleh seluruh pelaku ekonomi dalam satu tahun",
            "explanation": "Pendekatan pengeluaran menghitung seluruh pengeluaran dari rumah tangga, perusahaan, pemerintah, dan sektor luar negeri."
        },
        {
            "id": "C209",
            "question":"GDP Nominal suatu negara pada tahun tertentu tercatat lebih tinggi daripada GDP Riil. Untuk mengetahui pengaruh perubahan tingkat harga terhadap nilai produksi tersebut, pemerintah menggunakan GDP Deflator yang diperoleh dari ...",
            "options": [
                "A. Perbandingan GDP Riil dengan jumlah penduduk",
                "B. Selisih antara GDP Nominal dan GDP Riil",
                "C. Perbandingan GDP Nominal terhadap GDP Riil yang dikalikan 100",
                "D. Penjumlahan GDP Nominal dan GDP Riil",
                "E. Perbandingan GDP Riil terhadap GDP Nominal yang dikalikan 100"
            ],
            "answer":"C. Perbandingan GDP Nominal terhadap GDP Riil yang dikalikan 100",
            "explanation":"GDP Deflator digunakan untuk mengukur perubahan tingkat harga dalam perekonomian. Nilainya diperoleh dengan membandingkan GDP Nominal terhadap GDP Riil kemudian dikalikan 100."
        }
        
    ],

    "C3": [
        {
            "id": "C301",
            "question": """Perhatikan data perekonomian berikut (dalam miliar rupiah)
            -	GDP: Rp500.000 
            -	Pendapatan WNA di dalam negeri: Rp45.000 
            -	Pendapatan WNI di luar negeri: Rp30.000 
            -	Depresiasi: Rp15.000 
            -	Pajak Tidak Langsung: Rp22.000 
            -	Subsidi: Rp7.000 
            -	Laba Ditahan: Rp8.000 
            -	Iuran Asuransi: Rp4.000 
            -	Pajak Perseroan: Rp3.000 
            -	Transfer Payment: Rp12.000 
            -	Pajak Langsung: Rp14.000 
            Besar nilai Personal Income yang tepat berdasarkan data di atas adalah...
            """,
            "options": [
                "A. Rp485.000 miliar",
                "B. Rp470.000 miliar",
                "C. Rp462.000 miliar",
                "D. Rp452.000 miliar",
                "E. Rp445.000 miliar"

            ],
            "answer": "D. Rp452.000 miliar",
            "explanation":"""Penyelesaian:
                •	GNP = GDP + Pendapatan WNI di Luar Negeri - Pendapatan WNA di dalam negeri.
                •	GNP = Rp500.000 + Rp30.000 - Rp45.000
                •	GNP = Rp485.000
                •	NNP = GNP - Penyusutan
                •	NNP = Rp485.000 - Rp15.000
                •	NNP = Rp470.000
                •	NNI = NNP - Pajak Tidak Langsung + Subsidi
                •	NNI = Rp470.000 - Rp22.000 + Rp7.000
                •	NNI = Rp455.000
                •	PI = (NNI + Transfer Payment)- (Laba ditahan + Iuran asuransi + pajak perseroan)
                •	PI = (Rp455.000 + Rp12.000)- (Rp8.000 + Rp4.000 + Rp3.000)
                •	PI = Rp467.000 - Rp15.000
                •	PI = Rp452.000
                """
        },
        {
        "id": "C302",
        "question": "Diketahui alur produksi pakaian sebagai berikut: Petani menjual kapas Rp10.000; Pabrik memintal kapas jadi benang seharga Rp25.000; Pabrik kain menenun benang jadi kain seharga Rp60.000; Garmen menjahit kain jadi kemeja seharga Rp100.000. Berapakah total pendapatan nasional jika dihitung dengan menjumlahkan seluruh Nilai Tambah (Value Added)-nya?",
        "options": [
            "A. Rp195.000",
            "B. Rp100.000",
            "C. Rp60.000",
            "D. Rp35.000",
            "E. Rp10.000"
        ],
        "answer": "B. Rp100.000",
        "explanation": "Jumlah seluruh nilai tambah sama dengan nilai barang akhir yaitu Rp100.000."
    },

    {
        "id": "C303",
        "question": "Berdasarkan data keuangan berikut (dalam miliar rupiah): Upah Rp450.000, Sewa Rp120.000, Bunga Neto Rp60.000, dan Laba Usaha Rp160.000. Hitunglah besar pendapatan nasional menggunakan pendekatan pendapatan!",
        "options": [
            "A. Rp500.000",
            "B. Rp680.000",
            "C. Rp790.000",
            "D. Rp840.000",
            "E. Rp1.340.000"
        ],
        "answer": "C. Rp790.000",
        "explanation": "Y = r + w + i + p = 120.000 + 450.000 + 60.000 + 160.000 = Rp790.000."
    }

    ],

    "C4": [
        {
        "id": "C401",
        "question": "Kurva Lorenz digunakan sebagai grafik untuk menunjukkan realitas distribusi pendapatan. Semakin melengkung ke bawah menjauhi garis diagonal, maka hal itu menandakan...",
        "options": [
            "A. Distribusi pendapatan di negara itu semakin merata sempurna",
            "B. Distribusi pendapatan di negara itu semakin timpang/tidak merata",
            "C. Pendapatan per kapita negara tersebut melonjak drastis",
            "D. Nilai inflasi negara tersebut berada di angka nol persen",
            "E. Jumlah penduduk miskin di negara tersebut berkurang"
        ],
        "answer": "B. Distribusi pendapatan di negara itu semakin timpang/tidak merata",
        "explanation": "Semakin jauh Kurva Lorenz dari garis diagonal, semakin tinggi tingkat ketimpangan distribusi pendapatan."
    }
    ],

    "C5": [
        {
        "id": "C501",
        "question": """Perhatikan data koefisien gini berikut!
        - Asgard = 0,28
        - Midgard = 0,47
        - Vanaheim = 0,66
        Berdasarkan data tersebut, evaluasi yang paling tepat mengenai kondisi distribusi pendapatan ketiga negara adalah ...
        """,
        "options": [
            "A. Negara Asgard memiliki tingkat ketimpangan tertinggi",
            "B. Negara Midgard memiliki distribusi pendapatan paling merata",
            "C. Negara Vanaheim memiliki tingkat ketimpangan tertinggi, sedangkan Negara Asgard paling merata",
            "D. Negara Asgard dan Negara Vanaheim memiliki tingkat ketimpangan yang sama",
            "E. Negara Midgard memiliki ketimpangan yang lebih tinggi daripada Negara Vanaheim"

        ],
        "answer": "C. Negara Vanaheim memiliki tingkat ketimpangan tertinggi, sedangkan Negara Asgard paling merata",
        "explanation": "Semakin besar nilai Koefisien Gini mendekati angka 1, semakin tinggi tingkat ketimpangan pendapatan. Sebaliknya, semakin mendekati angka 0, distribusi pendapatan semakin merata. Karena Negara Vanaheim memiliki nilai tertinggi (0,66), tingkat ketimpangannya paling tinggi, sedangkan Negara Asgard (0,28) memiliki distribusi pendapatan paling merata."
    }
    ]
}
    def generate_pdf(
        name,
        class_name,
        user_answers,
        all_questions,
        score,
        rekomendasi,
        level_count,
        prediksi
    ):

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)

        width, height = A4


        nama_level = {
            "C1": "Mengingat",
            "C2": "Memahami",
            "C3": "Menerapkan",
            "C4": "Menganalisis",
            "C5": "Mengevaluasi"
        }

        jumlah_benar = score
        jumlah_salah = len(all_questions) - score

        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(
            width/2,
            height-1.2*cm,
            "LAPORAN HASIL ANALISIS KEMAMPUAN SISWA"
        )

        c.setFont("Helvetica", 8)

        y = height - 2.2*cm


        c.rect(1*cm, y-1.1*cm, 18*cm, 1.2*cm)

        c.drawString(1.2*cm, y-0.3*cm, f"Nama")
        c.drawString(3.5*cm, y-0.3*cm, f": {name}")

        c.drawString(10*cm, y-0.3*cm, f"Kelas")
        c.drawString(12*cm, y-0.3*cm, f": {class_name}")

        c.drawString(1.2*cm, y-0.8*cm, f"Skor")
        c.drawString(3.5*cm, y-0.8*cm, f": {score}/{len(all_questions)}")

        c.drawString(10*cm, y-0.8*cm, f"Akurasi Jawaban")
        c.drawString(
            13.8*cm,
            y-0.8*cm,
            f": {(score/len(all_questions))*100:.1f}%"
        )

        y -= 1.8*cm


        c.setFont("Helvetica-Bold", 10)

        c.drawString(
            1*cm,
            y,
            "HASIL ANALISIS KEMAMPUAN"
        )

        y -= 0.4*cm

        c.line(
            1*cm,
            y,
            19*cm,
            y
        )

        y -= 0.5*cm

        c.setFont("Helvetica", 8)

        c.drawString(
            1.2*cm,
            y,
            f"Prediksi Kelemahan Utama : {prediksi} ({nama_level[prediksi]})"
        )

        y -= 0.7*cm

        c.drawString(
            1.2*cm,
            y,
            f"Level yang Perlu Diperkuat : {prediksi} ({nama_level[prediksi]})"
        )

        y -= 1*cm

        c.setFont("Helvetica-Bold", 9)

        c.drawString(
            1*cm,
            y,
            "RINGKASAN KESALAHAN BERDASARKAN LEVEL KOGNITIF"
        )

        y -= 0.4*cm

        c.line(
            1*cm,
            y,
            19*cm,
            y
        )

        y -= 0.7*cm

        c.setFont("Helvetica", 8)

        data_level = [
            ["C1", "Mengingat", level_count["C1"]],
            ["C2", "Memahami", level_count["C2"]],
            ["C3", "Menerapkan", level_count["C3"]],
            ["C4", "Menganalisis", level_count["C4"]],
            ["C5", "Mengevaluasi", level_count["C5"]]
        ]

        for row in data_level:

            c.drawString(1.5*cm, y, row[0])
            c.drawString(3.0*cm, y, row[1])
            c.drawString(8.0*cm, y, str(row[2]))
            c.drawString(9.0*cm, y, "kesalahan")

            y -= 0.5*cm


        y -= 0.5*cm

        c.setFont("Helvetica-Bold", 9)

        c.drawString(
            1*cm,
            y,
            "REKOMENDASI MATERI"
        )

        y -= 0.4*cm

        c.line(
            1*cm,
            y,
            19*cm,
            y
        )

        y -= 0.7*cm

        c.setFont("Helvetica", 8)

        for i, item in enumerate(rekomendasi[:10], start=1):

            c.drawString(
                1.5*cm,
                y,
                f"{i}. {item}"
            )

            y -= 0.45*cm


        y -= 0.5*cm

        c.setFont("Helvetica-Bold", 9)

        c.drawString(
            1*cm,
            y,
            "RINGKASAN HASIL JAWABAN"
        )

        y -= 0.4*cm

        c.line(
            1*cm,
            y,
            19*cm,
            y
        )

        y -= 0.6*cm

        c.setFont("Helvetica", 7)

        x1 = 1.5*cm
        x2 = 10.5*cm

        for i in range(len(all_questions)):

            status = "✓" if user_answers[i] == all_questions[i]["answer"] else "✗"

            if i < len(all_questions)/2:

                c.drawString(
                    x1,
                    y - (i*0.35*cm),
                    f"Soal {i+1} : {status}"
                )

            else:

                c.drawString(
                    x2,
                    y - ((i-len(all_questions)//2)*0.35*cm),
                    f"Soal {i+1} : {status}"
        )
    

        c.showPage()

        width, height = A4

        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(
            width/2,
            height-1*cm,
            "PEMBAHASAN HASIL LATIHAN SOAL"
        )

        y = height - 1.7*cm

        c.setFont("Helvetica", 8)

        for idx, q in enumerate(all_questions):

            status = (
                "BENAR"
                if user_answers[idx] == q["answer"]
                else "SALAH"
            )

            # kalau mepet bawah halaman
            if y < 2*cm:
                c.showPage()
                y = height - 1.5*cm
                c.setFont("Helvetica", 8)

            # nomor soal
            c.setFont("Helvetica-Bold", 8)

            c.drawString(
                1*cm,
                y,
                f"Soal {idx+1} [{status}]"
            )

            y -= 0.3*cm

            c.setFont("Helvetica", 8)

            # pertanyaan
            soal_lines = textwrap.wrap(
                q["question"],
                width=110
            )

            for line in soal_lines:

                c.drawString(
                    1.2*cm,
                    y,
                    line
                )

                y -= 0.28*cm

            # jawaban siswa
            c.drawString(
                1.2*cm,
                y,
                f"Jawaban Siswa : {user_answers[idx]}"
            )

            y -= 0.28*cm

            # jawaban benar
            c.drawString(
                1.2*cm,
                y,
                f"Jawaban Benar : {q['answer']}"
            )

            y -= 0.28*cm

            # pembahasan
            pembahasan_lines = textwrap.wrap(
                q["explanation"],
                width=120
            )

            for line in pembahasan_lines:

                c.drawString(
                    1.2*cm,
                    y,
                    line
                )

                y -= 0.28*cm

            # garis tipis
            c.line(
                1*cm,
                y,
                19*cm,
                y
            )

            y -= 0.25*cm

        c.save()

        buffer.seek(0)

        return buffer.getvalue()


    st.header("Soal-soal Latihan Pendapatan Nasional")

    all_questions = (
        questions["C1"] +
        questions["C2"] +
        questions["C3"] +
        questions["C4"] +
        questions["C5"]
    )

    user_answers = {}

    with st.form("quiz_form"):
        name = st.text_input("Nama Lengkap")
        class_name = st.text_input("Kelas")

        for idx, q in enumerate(all_questions):
            st.subheader(f"No. {idx + 1}")

            st.markdown(q["question"])

            user_answers[idx] = st.radio(
                "",
                q["options"],
                index=None,
                key=f"q{idx}",
                label_visibility="collapsed"
            )

        submitted = st.form_submit_button("Kirim Jawaban")

    if submitted:

        belum_dijawab = []

        for idx in range(len(all_questions)):
            if user_answers[idx] is None:
                belum_dijawab.append(idx + 1)

        if len(belum_dijawab) > 0:

            st.error(
                f"Masih ada soal yang belum dijawab, nih! Cek soal nomor:{', '.join(map(str, belum_dijawab))}"
            )

            st.warning(
                "Cek jawaban kamu sebelum mengirimkan jawaban, ya!."
            )

            st.stop()

        score = 0
        soal_salah = []

        for idx, q in enumerate(all_questions):

            if user_answers[idx] == q["answer"]:
                score += 1
            else:
               
                soal_salah.append(q["id"])

        if score <= 5:
            level = "tingkat pemahaman kamu masih berada di level rendah, nih. Yuk, terus semangat belajar!"
        elif score <= 10:
            level = "tingkat pemahaman kamu berada di level menengah, nih. Yuk, terus semangat belajar!"
        else:
            level = "tingkat pemahaman kamu berada di level tinggi, nih. Pertahankan dan terus semangat belajar, ya!"

        st.subheader("Lihat hasil penilaian di bawah, ya!")
        st.info(f"Skor kamu: {score} dari {len(all_questions)}")

        st.success(f"Saat ini, {level}")

        st.subheader("Pembahasan Soal")

        for idx, q in enumerate(all_questions):

            st.write(f"No. {idx + 1}: {q['question']}")

            if user_answers[idx] == q["answer"]:
                st.success(f"Jawaban Kamu: {user_answers[idx]} ✅")

            else:
                st.error(f"Jawaban Kamu: {user_answers[idx]} ❌")
                st.info(f"Jawaban Benar: {q['answer']}")
                st.markdown(f"Pembahasan: {q['explanation']}")

            st.divider()

        #REKOMENDASI MATERI DENGAN KNN
        rekomendasi = []

        try:

            materi = pd.read_csv(
                "Somat.csv",
                sep=";"
            )

            #MENGHITUNG JUMLAH SALAH TIAP LEVEL

            def hitung_level_kognitif(soal_salah):

                data_salah = materi[
                    materi["ID"].isin(soal_salah)
                ]

                level_count = {
                    "C1": 0,
                    "C2": 0,
                    "C3": 0,
                    "C4": 0,
                    "C5": 0
                }

                for level in data_salah["Level_Kognitif"]:

                    if level in level_count:
                        level_count[level] += 1

                return level_count

            #FUNGSI REKOMENDASI

            def rekomendasi_materi(soal_salah):

                if len(soal_salah) == 0:
                    return []

                level_count = hitung_level_kognitif(
                    soal_salah
                )
                
                profil_siswa = pd.DataFrame(
                    [[
                        level_count["C1"],
                        level_count["C2"],
                        level_count["C3"],
                        level_count["C4"],
                        level_count["C5"]
                    ]],
                    columns=["C1","C2","C3","C4","C5"]
                )

                #DATATRAINTEST KNN
        
                data = pd.read_csv("data.csv")

                X = data[["C1","C2","C3","C4","C5"]]
                y = data["Label"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42
                )

                for k in [1]:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)

                    print(f"k={k} -> {acc*100:.2f}%")

                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_train, y_train)

                prediksi = knn.predict(profil_siswa)[0]
                
                rekomendasi = materi[
                materi["Level_Kognitif"] == prediksi

                ]["Materi"]

                return rekomendasi[:10], prediksi


            level_count = hitung_level_kognitif(
                soal_salah
            )
            rekomendasi, prediksi = rekomendasi_materi(
                soal_salah
            )

            st.subheader(
                "Hasil Analisis Kemampuan Kamu:"
            )
            st.info(
                f"Kelemahan utama kamu ada di soal level {prediksi}, nih"
            )

            nama_level = {
                "C1": "Mengingat",
                "C2": "Memahami",
                "C3": "Menerapkan",
                "C4": "Menganalisis",
                "C5": "Mengevaluasi"
            }

            st.write(f"Level mengingat(C1) ada {level_count['C1']} kesalahan")
            st.write(f"Level memahami(C2) ada {level_count['C2']} kesalahan")
            st.write(f"Level menerapkan(C3) ada {level_count['C3']} kesalahan")
            st.write(f"Level menganalisis(C4) ada {level_count['C4']} kesalahan")
            st.write(f"Level mengevaluasi(C5) ada {level_count['C5']} kesalahan")
            
            st.warning(
                f"Level yang perlu kamu perkuat {prediksi} ({nama_level[prediksi]}), ya"
            )
            st.divider()
            st.subheader("Aku sudah siapkan rekomendasi materi yang bisa kamu pelajari lagi, nih...")
            if len(rekomendasi) > 0:
            
                for item in rekomendasi:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.success(
                    "Hebat! Saat ini kemampuan kamu sudah sangat baik. Pertahankan dan selalu semangat belajar ekonomi, ya!."
                )

        except Exception as e:

            st.error(
                f"Yahh:( sepertinya ada kesalahan saat memuat: {e}"
            )

        pdf_output = generate_pdf(
            name, 
            class_name, 
            user_answers, 
            all_questions, 
            score,
            rekomendasi,
            level_count,
            prediksi
        )
        
        st.download_button(
            label="Download Hasil Latihan (PDF)",
            data=pdf_output,
            file_name=f"Hasil Latihan {name}.pdf",
            mime="application/pdf"
        )

        if "hasil_latihan" not in st.session_state:
            st.session_state.hasil_latihan = None

        st.session_state.hasil_latihan = {
            "nama": name,
            "kelas": class_name,
            "score": score,
            "soal_salah": soal_salah,
            "rekomendasi": rekomendasi
        }

    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"

# FAKTA MENARIK
elif st.session_state.active_tab == "tab9":
    st.header("Fakta-fakta menarik yang harus kamu ketahui tentang Pendapatan Nasional, nih!")
    st.subheader("Fakta 1")
    st.image('ue.jpg')
    st.info("""
        **Tahukah kamu?**
            

    Kegiatan ekonomi informal tidak selalu tercatat dalam GDP, lho.
    Di beberapa negara berkembang, kegiatan ekonomi informal ini bisa menyumbang lebih dari 40% dari ekonomi. Contohnya, pedagang kaki lima,
    buruh lepas, atau usaha rumahan yang tidak memiliki izin usaha. Usaha-usaha tersebut bisa menyediakan lapangan kerja dan membantu memenuhi kebutuhan sehari-hari banyak orang.
    Fenomena ini disebut "Underground Economy" atau "Ekonomi Bawah Tanah". 
        
    Kegiatan informal ini banyak ditemukan di beberapa negara berkembang, seperti India, Nigeria, Kenya, Filipina, dan negara kita tercinta, Indonesia.
  
            
    Teman-teman bayangkan saja. kalau ekonomi informal bisa sepenuhnya tercatat, dampaknya pasti luar biasa. Angka GDP bisa melonjak!
    Pemerintah juga dapat pendapatan pajak lebih banyak, yang artinya ada lebih banyak dana untuk membangun infrastruktur,
    memperbaiki layanan kesehatan, dan meningkatkan pendidikan. Tidak cuma itu, pekerja yang sebelumnya tidak terlindungi akan punya akses ke jaminan sosial, gaji yang layak, dan perlindungan hukum.
    Usaha kecil yang tadinya sulit berkembang juga bisa lebih mudah dapat pinjaman atau bantuan.
    Bayangkan saja kios kecil di pinggir jalan, yang selama ini hanya bertahan dari hari ke hari, bisa punya peluang untuk tumbuh jadi bisnis besar.       
    """)

    st.subheader("Fakta 2")
    st.image('ntl..png')
    st.info("""
    **Tahukah kamu?**
            

    Ada cara unik, lho, untuk mengukur pendapatan nasional di suatu negara, yaitu menggunakan pencahayaan malam hari yang ditangkap satelit!.
    Cara ini menggunakan sensor Nighttime Light (NTL) atau Pencahayaan Malam.
            
    
    Dilansir dari Mapid.co.id, salah satu indikator tingkat perekonomian yang baik dalam suatu wilayah dapat dikenali dari tingkat aktivitas komersial pada malam hari,
    yang tercermin dalam data NTL. Data ini dapat memberikan analisis komparasi wilayah yang memiliki tingkat aktivitas ekonomi yang tinggi dan rendah pada waktu tertentu. Wilayah dengan cahaya malam yang
    lebih terang seringkali menunjukkan pusat-pusat kegiatan ekonomi yang kuat,
    sementara wilayah dengan cahaya malam yang lebih redup cenderung memiliki aktivitas ekonomi yang rendah.
    Bagaimana, menarik banget bukan?
    """)

    st.subheader("Fakta 3")
    st.image('greengdp.jpg')
    st.info("""
    **Tahukah kamu?**
            

    Terdapat cara baru, lho, dalam menghitung GDP, yaitu menggunakan konsep "Green GDP" atau "GDP Hijau".
    
            
    Berdasarkan hasil webinar "The Quest for Green GDP" yang dilaksanakan pada 20 April 2023 oleh Divisi Statistik Perserikatan Bangsa-Bangsa dan Bank Dunia,
    GDP biasa hanya mengukur nilai moneter barang dan jasa yang diproduksi dalam setahun untuk mengukur pertumbuhan ekonomi, tetapi pertumbuhan ekonomi yang terjadi dengan mengorbankan alam justru bisa merugikan masa depan.
    Green GDP diperkenalkan pada akhir 1980-an untuk memperhitungkan dampak negatif ekonomi terhadap lingkungan.
    Intinya, konsep Green GDP yaitu cara mengukur GDP dengan menghitung dampak lingkungan dari aktivitas ekonomi. Jika suatu aktivitas membuat kerusakan lingkungan (seperti polusi), nilai GDPnya bisa dikurangi.
    """)

    st.subheader("Fakta 4")
    st.image('pp.jpg')
    st.info("""
    **Tahukah kamu?**

            
    Tahun 2016, terdapat dokumen rahasia yang bocor, bernama Panama Papers. Berdasarkan buku 'The Panama Papers: Breaking Story of How the Rich & Powerful Hide Their Money' yang ditulis oleh Bastian Obermayer dan Frederik Obermaier, bahwa Panama Papers ini
    pertama kali disampaikan kepada jurnalis oleh seorang whistleblower anonim yang menggunakan nama samaran John Doe. Melalui saluran komunikasi terenkripsi, John Doe
    menghubungi Bastian Obermayer (Penulis buku terseut yang adalah seorang jurnalis) dari Suddeutzche Zeitung, dan menawarkan akses ke sejumlah besar dokumen yang berasal dari firma
    hukum Mossac Fonseca. Dokumen-dokumen tersebut berisi rincian perusahaan offshore, pemiliknya, dan bagaimana kekayaan mereka disembunyikan dari otoritas pajak.
            
    Perusahaan offshore atau perusahaan cangkang adalah perusahaan yang didirikan di luar negara tempat pemiliknya tinggal atau beroperasi. Biasanya, perusahaan ini didirikan di negara
    yang memiliki regulasi perpajakan yang lebih menguntunhkan atau kebijakan bisnis yang lebih longgar. Mossack Fonseca adalah sebuah firma hukum yang berperan sebagai perantara utama dalam membantu individu atau perusahaan untuk mendirikan perusahaan offshore.
    Mereka menyediakan layanan untuk membuat dan mengelola perusahaan cangkang yang memungkinkan klien untuk menyembunyikan kekayaan dan menghindari pajak. Mozzack Fonseca juga mengurus berbagai dokumen dan izin yang diperlukan, sehingga klien mereka bisa beroperasi di luar jangkaun hukum
    negara mereka. Hal ini termasuk mengatur rekening bank di negara-negara yang memiliki kebijakan pajak rendah atau tidak ada pajak sama sekali. Jadi, mereka menjadi kunci dalam jaringan penghindaran pajak global yang terungkap dalam skandal Panama Papers.
            
    Awal mula pengungkapan Panama Papers ini melibatkan sekelompok jurnalis investigasi, terutama dari Suudeutzche Zeitung (salah satu surat kabar harian terbesar dan terkemuka di Jerman), yang dipimpin oleh Bastian Obermayer dan Frederik Obermaier. Investigasi terhadap dokumen ini
    melibatkan kolaborasi antara jurnalis dari berbagai negara yang dipimpin oleh International Consartium of Investigative Jounalists (ICIJ).
    Berikut adalah isi dari Panama Papers yang terungkap:
    - Total dokumen yang terekspos sebanyak 11,5 juta dokumen.
    - Nama-nama perusahaan, alamat, dan detail pendaftaran 214.000 perusahaan offshore yang didirikan oleh Mossac Fonseca.
    - Nama-nama individu atau entitas yang memiliki atau terlibat dengan perusahaan-perusahaan tersebut.
    - Rekening bank, laporan keuangan, dan transaksi keuangan.
    - Data dari ICIJ, ada 2.961 nama warga negara Indonesia yang terdaftar pada dokumen tersebut.

            
    Panama Papers ini sangat erat kaitannya dengan konsep Tax Heaven atau Surga Pajak. Tax Heaven adalah negara atau wilayah yang memiliki kebijakan pajak yang sangat rendah atau bahkan tidak ada pajak yang ditarik. Padahal, untuk negara yang sangat mengandalkan pajak sebagai pendapatan negara pasti sangat dirugikan.
    Indonesia misalnya, berdasarkan data dari Badan Pusat Statistika (BPS), pendapatan pajak Indonesia menyentuh 82,4% dari pendapatan nasional secara keseluruhan. Gede banget, kan, angkanya!.
    Oleh karena itu, Tax Heaven ini sangat merugikan negara, karena memunginkan individu atau perusahaan kaya untuk menghindari kewajiban membayar pajak mereka di negara asal.
            
    Orang kaya dapat menyimpan uang mereka di Tax Heaven dan menghindari pajak, sementara masyarakat biasa yang tidak memiliki akses ke cara-cara tersebut tetap membayar pajak mereka dengan susah payah. 'Nggak adil banget, kan?
    """)

    st.subheader("Fakta 5")
    st.image('simon.jpg')
    st.info("""
    Tahukah kamu?
    Pencipta formula PDB modern sendiri justru membenci ciptaannya jika digunakan untuk mengukur kesejahteraan? Simon Kuznets, ekonom yang mengembangkan konsep ini pada tahun 1934 untuk Pemerintah AS, secara tegas memperingatkan dalam laporan pertamanya ke Kongres:
    
    "Kesejahteraan suatu bangsa hampir tidak dapat diukur dari tingkat pendapatan nasionalnya."
            
    Kuznets mengingatkan bahwa PDB itu buta. PDB akan melonjak naik jika terjadi perang (karena produksi senjata meningkat), polusi masif (karena biaya pembersihan menghasilkan uang), atau wabah penyakit
    (karena belanja kesehatan meroket). Sebaliknya, pekerjaan mulia yang tidak digaji seperti ibu rumah tangga yang mendidik anaknya atau merawat lansia bernilai nol besar di mata PDB.

    """)

    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0"
  
# REFERENSI BACAAN
elif st.session_state.active_tab == "tab10":
    st.header("Referensi bacaan materi Pendapatan Nasional")
    st.markdown("""
        Kamu bisa memperdalam pengetahuan kamu tentang Pendapatan Nasional dengan membaca referensi berikut ini:
        
        - E-Modul Ekonomi XI, Kemendikbud 2019, Penyusun: Wahyu Rini Mulyasari, S.Pd.
        - Modul Pembelajaran SMA Ekonomi XI, Kemendikbud 2020, Penyusun: Anna Monalita de Fretes, S.Pd., M.Pd.
        """)
    if st.button("Kembali ke Menu Utama"):
        st.session_state.active_tab = "tab0" 
