# Model
import torch
# Image processing
import matplotlib.pyplot as plt
import cv2


def predict_image(img, model, device):
    # Convert to a batch of 1
    xb = img.unsqueeze(0).to(device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return preds


def face_detection(path, model, my_transforms):
    image = cv2.imread(path)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=4)
    for x, y, w, h in faces:
        only_face_gray = image_gray[y:y + h, x:x + w]
        only_face = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        facess = faceCascade.detectMultiScale(only_face_gray)
        if len(facess) == 0:
            print("Face not detected!")
        else:
            for (ex, ey, ew, eh) in facess:
                cropped_face = only_face[ey:ey + eh, ex:ex + ew]

            final_image = my_transforms(cropped_face)

            cropped_face = cv2.resize(cropped_face, (224, 224))

            font = cv2.FONT_HERSHEY_PLAIN  # font to put text on the image

            prediction = predict_image(final_image, model)
            print(prediction)
            if prediction == 0:
                print(f"With mask")
                text = f"mask"
                cv2.putText(cropped_face, text, (10, 200), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print(f"Without mask")
                text = f"no mask"
                cv2.putText(cropped_face, text, (10, 200), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Plot two images
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.title('Full image')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 2, 2)
            plt.title('face_only')
            plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
