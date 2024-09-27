import face_recognition
import cv2
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Configureer de logger
def configure_logger(log_directory='log', filename='log.txt', level=logging.DEBUG):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(os.path.join(log_directory, filename), mode='w')  # Overwrites the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional: Add a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def log_message(message, level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.log(level, message)

# Laad de personen uit personen.txt
def load_personen(file_path):
    log_message(f"Probeer het bestand '{file_path}' te openen.", level=logging.INFO)
    with open(file_path) as file:
        personen = [line.strip() for line in file]
    log_message(f"Aantal personen geladen: {len(personen)}.", level=logging.INFO)
    return personen

# Laad gezichtafbeeldingen en koppel ze aan namen
def load_known_faces(directory, personen):
    known_face_encodings = []
    known_face_names = []
    
    for persoon in personen:
        # Splits de naam in voor- en achternaam
        voornaam, achternaam = persoon.split()
        image_path = os.path.join(directory, f"{voornaam}_{achternaam}.jpg")
        
        if not os.path.exists(image_path):
            log_message(f"Afbeelding '{image_path}' niet gevonden.", level=logging.WARNING)
            continue
        
        face_image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face_image)
        
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(persoon)
            log_message(f"Face encoding geladen voor '{persoon}' van '{image_path}'", level=logging.INFO)
        else:
            log_message(f"Geen gezicht encoding gevonden voor '{image_path}'", level=logging.WARNING)

    return known_face_encodings, known_face_names

# Verkrijg het huidige werkdirectory
base_directory = os.getcwd()
personen_file = os.path.join(base_directory, "personendata", "personen.txt")

# Laad de personen
personen = load_personen(personen_file)

# Specificeer de directory voor de gezichtafbeeldingen
face_directory = os.path.join(base_directory, "personendata")
known_face_encodings, known_face_names = load_known_faces(face_directory, personen)

# Registratie log configuratie
today = datetime.now().strftime('%Y-%m-%d')
registration_log_path = os.path.join('log', f'registratie_{today}.log')
registrations = {}

# Webcam configuratie
video_capture = cv2.VideoCapture(0)  # Gebruik default camera
if not video_capture.isOpened():
    log_message("Webcam niet gevonden! Controleer de verbinding.", level=logging.ERROR)
    exit()

# Drempelwaarde voor registratie
confidence_threshold = 60.0  # Drempel ingesteld op 60%

# Programmaloop
while True:
    ret, frame = video_capture.read()
    if not ret:
        log_message("Frame niet goed gelezen.", level=logging.ERROR)
        continue

    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Onbekend"
        registration_time = None
        confidence = None

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = (1 - face_distances[best_match_index]) * 100  # Zet het in procenten

            if matches[best_match_index] and confidence >= confidence_threshold:
                name = known_face_names[best_match_index]
                log_message(f"Herkenning van gezicht: {name} met zekerheid: {confidence:.2f}%")

                # Controleer registratie tijd
                now = datetime.now()
                if name not in registrations or (now - registrations[name]) > timedelta(hours=2):
                    with open(registration_log_path, 'a') as reg_file:
                        reg_file.write(f"{now} - Gezicht geregistreerd: {name} met zekerheid: {confidence:.2f}%\n")
                    registrations[name] = now
                    registration_time = now.strftime('%Y-%m-%d %H:%M:%S')
                    cv2.putText(frame, "Registratie opgeslagen", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Registratie recent!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            else:
                log_message(f"Geen geldige registratie voor: {name} met zekerheid: {confidence:.2f}%", level=logging.WARNING)

        face_names.append(name)

    # Resultaten weergeven
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        # Toon registratietijd als geregistreerd
        if name in registrations:
            registration_time = registrations[name].strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, f"Geregistreerd: {registration_time}", (left, bottom + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        # Toon zekerheid percentage
        if confidence is not None:
            cv2.putText(frame, f"Zekerheid: {confidence:.2f}%", (left, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Afsluiten
video_capture.release()
cv2.destroyAllWindows()
