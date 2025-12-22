import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Found solutions via direct import")
    print(solutions.face_detection)
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
