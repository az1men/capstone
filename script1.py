# %% Импорт библиотек
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

print("Script is running...")

ultrasonic = DistanceSensor(echo=17, trigger=4)
LED_PIN = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)



# Пути к модели и классы
MODEL_PATH = "best_model.pth"
CLASSES = ["Black Scurf", "Blackleg", "Common Scab", "Dry Rot", "Healthy Potatoes", "Miscellaneous", "Pink Rot"]

# %% Загрузка модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50()  # Загружаем архитектуру ResNet50
model.fc = torch.nn.Linear(in_features=2048, out_features=len(CLASSES))  # Подстраиваем под число классов
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Загружаем веса
model.to(DEVICE)
model.eval()

print("✅ Модель успешно загружена!")

# %% Преобразование изображения (то же самое, что при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %% Открытие камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Не удалось получить кадр")
        break

    # Отображаем видео с камеры
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    #if key == ord('c'):  # Нажатие 'c' для снимка
    #    print("📸 Снимок сделан. Определяем...")

    distance_cm = ultrasonic.distance * 100
    if distance_cm < 40:  # If object is closer than 40 cm
        print("📸 Object detected! Taking a picture...")
        time.sleep(N) #wait N seconds for potato to reach camera

        # Преобразование изображения
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Перевод в RGB
        img = Image.fromarray(img)  # Преобразуем в PIL
        img = transform(img).unsqueeze(0).to(DEVICE)  # Применяем трансформации и добавляем batch size

        # Предсказание модели
        with torch.no_grad():
            predictions = model(img)
            class_idx = torch.argmax(predictions, dim=1).item()
            probability = torch.nn.functional.softmax(predictions, dim=1)[0][class_idx].item()

        class_label = CLASSES[class_idx]  # Получаем имя класса
        status = "Good" if class_label == "Healthy Potatoes" else "Bad"  # Определяем статус

        print(f"✅ Предсказание: {class_label} ({status}) с вероятностью {probability:.4f}")
        
        if status == "Bad":
            time.sleep(N) #wait N seconds for potato to reach linear actuator
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(2)  # Keep LED on for 2 seconds
            GPIO.output(LED_PIN, GPIO.LOW) 
    

    elif key == ord('q'):  # Выход по 'q'
        break

# Освобождаем ресурсы
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()