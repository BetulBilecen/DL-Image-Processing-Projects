"""
PROJE: GAN (Generative Adversarial Network) ile Moda Ürünü Tasarımı
VERİ SETİ: Fashion MNIST

Fashion MNIST veri seti 10 class içeren 28x28 boyutunda gri tonlamalı görüntülerden oluşur:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot
"""

# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import os
from keras.datasets import fashion_mnist

# ----------------------------------------------------------------------------
# 2. GLOBAL PARAMETRELER VE AYARLAR (GLOBAL CONFIGURATION)
# ----------------------------------------------------------------------------
BUFFER_SIZE = 60000  # veri seti boyutu
BATCH_SIZE  = 128    # 128 resim 1 paket olacak
EPOCHS      = 100
NOISE_DIM   = 100    # generator'a verilecek gürültü vektörünün boyutu
IMG_SHAPE   = (28, 28, 1)  # input boyutu

# ----------------------------------------------------------------------------
# 3. VERİ SETİNİN YÜKLENMESİ VE ÖN İŞLEME (DATA LOADING & PREPROCESSING)
# ----------------------------------------------------------------------------
(train_images, _), (_, _) = fashion_mnist.load_data()  # görüntüler alındı, etiketler kullanılmadı
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")  # şekillendirip float veri türüne çevrildi
train_images = (train_images - 127.5) / 127.5  # [-1, 1] arası normalizasyon
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ----------------------------------------------------------------------------
# 4. GENERATOR MODELİ (FAKE GÖRÜNTÜ ÜRETİMİ)
# ----------------------------------------------------------------------------
def Make_Generator_Model():
    model = keras.Sequential([
        # İlk tam bağlı katman: gürültüyü özellik haritasına çevirir
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),  # eğitim stabilitesini artırır
        layers.LeakyReLU(),           # negatif girişleri yumuşatır

        layers.Reshape((7, 7, 256)),  # boyut 3'e çıkarıldı: (7, 7, 256)

        # 7x7 → 7x7
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 7x7 → 14x14
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 14x14 → 28x28 — tanh kullanılmasının nedeni: normalizasyonda [-1,1] yaptık, tanh de bu aralıkta sıkıştırır
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")
    ])
    return model

# ----------------------------------------------------------------------------
# 5. DİSCRİMİNATOR MODELİ (GERÇEK / SAHTE AYIRIMI)
# ----------------------------------------------------------------------------
def Make_Discriminator_Model():
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=IMG_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),  # düzleştirme
        layers.Dense(1)    # sahte/gerçek şeklinde ikili sınıflandırma
    ])
    return model

# ----------------------------------------------------------------------------
# 6. KAYIP FONKSİYONLARI (LOSS FUNCTIONS)
# ----------------------------------------------------------------------------
cross_entropy = keras.losses.BinaryCrossentropy()

def Discriminator_Loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)   # gerçek → 1 etiketi atandı
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # sahte  → 0 etiketi atandı
    return real_loss + fake_loss  # toplam discriminator kaybı

def Generator_Loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # sahte görüntüyü 1 gibi göster

# ----------------------------------------------------------------------------
# 7. OPTİMİZATÖR TANIMLAMALARI (OPTIMIZERS)
# ----------------------------------------------------------------------------
generator_optimizer     = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ----------------------------------------------------------------------------
# 8. YARDIMCI FONKSİYONLAR (HELPER FUNCTIONS)
# ----------------------------------------------------------------------------
seed = tf.random.normal([16, NOISE_DIM])  # sabit gürültü örneği — her epoch'ta aynı seed ile görüntü üretilir

def Generate_And_Save_Images(model, epoch, test_input):
    predictions = model(test_input, training=False)  # modeli sadece değerlendirme modunda çalıştır
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap="gray")  # görüntüleri [0, 1] aralığına rescale et
        plt.axis("off")

    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")

    plt.savefig(f"generated_images/image_at_epoch{epoch:03d}.png")
    plt.close()

# ----------------------------------------------------------------------------
# 9. EĞİTİM FONKSİYONU (TRAINING)
# ----------------------------------------------------------------------------
def Train(dataset, epochs):
    for epoch in range(1, epochs + 1):
        gen_loss_total  = 0  # generator toplam kaybı
        disc_loss_total = 0  # discriminator toplam kaybı
        batch_count     = 0

        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])  # gürültü üretildi

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)             # sahte görüntüler üret

                real_output = discriminator(image_batch, training=True)        # gerçek görüntü sonucu
                fake_output = discriminator(generated_images, training=True)   # sahte görüntü sonucu

                gen_loss  = Generator_Loss(fake_output)                        # generator kaybı
                disc_loss = Discriminator_Loss(real_output, fake_output)       # discriminator kaybı

            # Gradyan hesabı
            gradients_gen  = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            # Gradyan güncelleme
            generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

            gen_loss_total  += gen_loss
            disc_loss_total += disc_loss
            batch_count     += 1

        # Her epoch sonunda kayıpları yazdır ve görüntü kaydet
        print(f"Epoch: {epoch}/{epochs} | Generator Loss: {gen_loss_total/batch_count:.3f} | Discriminator Loss: {disc_loss_total/batch_count:.3f}")
        Generate_And_Save_Images(generator, epoch, seed)

# ----------------------------------------------------------------------------
# 10. MODELLERİ OLUŞTUR VE EĞİTİMİ BAŞLAT
# ----------------------------------------------------------------------------
generator     = Make_Generator_Model()
discriminator = Make_Discriminator_Model()

Train(train_dataset, EPOCHS)