import face_recognition
import cv2

# Load images of the persons you want to recognize
known_image1 = face_recognition.load_image_file("WhatsApp Image 2024-06-04 at 2.03.42 PM (2).jpg")
known_encoding1 = face_recognition.face_encodings(known_image1)[0]

known_image2 = face_recognition.load_image_file("WIN_20240129_05_50_23_Pro (1).jpg")
known_encoding2 = face_recognition.face_encodings(known_image2)[0]

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known person
        matches1 = face_recognition.compare_faces([known_encoding1], face_encoding)
        matches2 = face_recognition.compare_faces([known_encoding2], face_encoding)

        name = "UNKNOWN"

        # If a match is found, use the name of the known person
        if matches1[0]:
            name = "HITESH"
        elif matches2[0]:
            name = "VAIBHAV"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
