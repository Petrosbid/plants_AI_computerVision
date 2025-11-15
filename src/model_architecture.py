# src/model_architecture.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B3  # می‌توانید مدل را تغییر دهید


def build_model(config, num_classes):
    """
    مدل SOTA را بر اساس تنظیمات می‌سازد (مرحله اول: فریز شده).
    """
    input_shape = tuple(config['input_shape'])

    # 1. ورودی
    inputs = Input(shape=input_shape, name="input_layer")

    # 2. مدل پایه (Base Model)
    # EfficientNetV2 ورودی‌های [0, 255] را انتظار دارد
    base_model = EfficientNetV2B3(
        include_top=config['include_top'],
        weights=config['weights'],
        input_tensor=inputs,
        pooling=config['pooling']
    )

    # فریز کردن مدل پایه
    base_model.trainable = False

    # 3. "سر" جدید (New Head)
    x = base_model.output
    x = Dropout(config['dropout_1'], name="top_dropout_1")(x)
    x = Dense(config['dense_1'], activation="relu", name="top_dense_1")(x)
    x = Dropout(config['dropout_2'], name="top_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="output_layer")(x)

    # 4. ساخت مدل نهایی
    model = Model(inputs=inputs, outputs=outputs, name=config['model_type'])

    print(f"Model built successfully with {num_classes} classes.")
    model.summary()

    return model


def compile_model_for_transfer_learning(model, config):
    """
    مدل را برای فاز اول (آموزش Head) کامپایل می‌کند.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_head']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled for Transfer Learning (Head Training).")
    return model


def compile_model_for_fine_tuning(model, config):
    """
    مدل را برای فاز دوم (Fine-Tuning) آماده و کامپایل می‌کند.
    """
    # باز کردن قفل لایه‌های مدل پایه
    base_model = model.get_layer(index=1)  # (index 1 معمولا مدل پایه است)
    base_model.trainable = True

    # فریز کردن لایه‌های پایین‌تر
    fine_tune_at = config['fine_tune_at_layer']
    if fine_tune_at > 0:
        print(f"Freezing all layers except the top {fine_tune_at} layers.")
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
    else:
        print("Unfreezing all layers for full fine-tuning.")

    # کامپایل مجدد با نرخ یادگیری بسیار پایین
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_fine_tune']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled for Fine-Tuning.")
    model.summary()  # نمایش وضعیت جدید trainable
    return model