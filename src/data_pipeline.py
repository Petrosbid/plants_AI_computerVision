# src/data_pipeline.py

import tensorflow as tf


def get_data_augmentation_layer(img_height, img_width):
    """
    یک لایه پیش‌پردازش برای Augmentation روی GPU می‌سازد.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def build_data_pipeline(config):
    """
    خط لوله داده آموزشی و اعتبارسنجی را با tf.data می‌سازد.

    Args:
        config (dict): دیکشنری خوانده شده از params.yaml

    Returns:
        tuple: (train_ds, val_ds, class_names)
    """
    data_dir = config['data_dir']
    img_height = config['input_shape'][0]
    img_width = config['input_shape'][1]
    batch_size = config['batch_size']
    val_split = config['validation_split']
    seed = config['seed']

    # استفاده از بافر اتوماتیک tf.data
    AUTOTUNE = tf.data.AUTOTUNE

    print(f"Loading training data from: {data_dir}")
    # ساخت دیتاست آموزشی
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    print(f"Loading validation data from: {data_dir}")
    # ساخت دیتاست اعتبارسنجی
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes.")

    # ساخت لایه Augmentation
    augmentation_layer = get_data_augmentation_layer(img_height, img_width)

    # اعمال Augmentation فقط به دیتاست آموزشی
    train_ds = train_ds.map(lambda x, y: (augmentation_layer(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)

    # پیش‌پردازش مدل پایه (EfficientNet خودش نرمال‌سازی 0-255 را انجام می‌دهد)
    # اگر از مدلی مانند ResNet استفاده می‌کنید، لایه Rescaling را اضافه کنید
    # preprocessor = tf.keras.applications.efficientnet_v2.preprocess_input
    # train_ds = train_ds.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE)
    # val_ds = val_ds.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE)

    # بهینه‌سازی عملکرد خط لوله
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names