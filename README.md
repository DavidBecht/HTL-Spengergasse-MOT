# Numpy IOU Tracker Aufgabe
## Beschreibung
Diese Übung befasst sich mit dem Tracking im Object Detection Umfeld. Tracking ist die
kontinuierliche Verfolgung von Objekten über mehrere Frames in einem Videostream oder einer 
Bildsequenz. Object Detection ist der Prozess, bei dem in einem Bild oder Frame Objekte 
identifiziert und begrenzende Rechtecke um sie herumgezogen werden. Tracking zielt darauf 
ab, diese erkannten Objekte im Laufe der Zeit zu verfolgen, um deren Position und Bewegung 
zu bestimmen.

<img src="img_1.png" alt="drawing" width="400px"/>
In der Abbildung von oben können Sie eine bildliche Beschreibung der Aufgabe sehen. Es sind verschiedene Objekte 
(O<sub>1</sub>, O<sub>2</sub>, O<sub>3</sub>) und Tracks (h<sub>1</sub>, h<sub>2</sub>, h<sub>3</sub>) zu erkennen. Vom
Trackingalgorithmus wird Objekt O<sub>1</sub> zum Track h<sub>1</sub> mit der Farbe blau zugeordnet über die Zeit 
(t, t+1, t+2, ...) zugeordnet. Beim Zeitpunkt t+3 kommt es zu einem Msmatch, dass heißt O<sub>1</sub> blau wird fälschlicherweise zu O<sub>2</sub> grün.

## Abgabe
Abzugeben ist diese Aufgabe via Moodle. Bitte löschen Sie vor der Abgabe alle Ausgaben: 
Cell → All Outputs → Clear. Alle Abgaben werden mit einem Plagiats-Checker überprüft 
werden. Wird ein Plagiat erkannt, werden alle beteiligten Personen für diese Aufgabe negativ beurteilt. Des Weiteren werden Sie Teilaufgaben der restlichen Klasse präsentieren.

## Beurteilung
Eine positive Beurteilung kann folgendermaßen erreicht werden:

- Anzahl der Punkte müssen über 50% der Gesamtpunkte sein
- In jeder Teilaufgabe mindestens eine Aufgabe gelöst
- Bonuspunkte für bessere Ergebnisse als der angeführte Mindeststandard
- Muss einer visuellen Einschätzung genügen
- Die Genauigkeit ihrer Methode muss diesen Mindeststandard einhalten (dh.: besser sein):

    | num_frames | idf1 | idp | idr | recall | precision | num_objects | mostly_tracked | partially_tracked |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 795 | 0.289712 | 0.25986 | 0.327312 | 0.905806 | 0.719139 | 4650 | 18 | 1 |

    | mostly_lost | num_false_positives | num_misses | num_switches | num_fragmentations | mota | motp |
    |-------------| --- | --- | --- | --- | --- | --- |
    | 0 | 1645 | 438 | 225 | 147 | 0.503656 | 0.292089 |

    | Name                 | Description                                                                        |
    | :------------------- | :--------------------------------------------------------------------------------- |
    | num_frames           | Total number of frames.                                                            |
    | num_switches         | Total number of track switches.                                                    |
    | num_false_positives  | Total number of false positives (false-alarms).                                    |
    | num_misses           | Total number of misses.                                                            |
    | mostly_tracked       | Number of objects tracked for at least 80 percent of lifespan.                     |
    | partially_tracked    | Number of objects tracked between 20 and 80 percent of lifespan.                   |
    | mostly_lost          | Number of objects tracked less than 20 percent of lifespan.                        |
    | num_fragmentations   | Total number of switches from tracked to not tracked.                              |
    | motp                 | Multiple object tracker precision.                                                 |
    | mota                 | Multiple object tracker accuracy.                                                  |
    | precision            | Number of detected objects over sum of detected and false positives.               |
    | recall               | Number of detections over number of objects.                                       |
    | idp                  | ID measures: global min-cost precision.                                            |
    | idr                  | ID measures: global min-cost recall.                                               |
    | idf1                 | ID measures: global min-cost F1 score.                                             |

## Datensatz
Der Datensatz befindet sich in dem Ordner `dataset` und ist folgendermaßen aufgeteilt: 
```
.
|—— <dataset>
	|—— gt
		|—— gt.txt
	|—— det
		|—— det.txt
	|—— img1
		|—— 000001.jpg
		|—— 000002.jpg
		|—— ….
```
### det.txt
Dieses File enthält die Detektionen über welche man einen Zusammenhang über die Zeit bestimmen muss. Diese 
Detektionen kommen im Normalfall von einem Deeplearning Netz, welches in jedem Frame nach Objekten sucht und diese mit
Bounding Boxes (Rechtecke) umrahmt.

#### Datenformat
Jede Zeile des Files enthält ein Objekt mit folgenden Werten:
<code>
&lt;frame&gt;,
&lt;id&gt;,
&lt;bb_left&gt;,
&lt;bb_top&gt;,
&lt;bb_width&gt;,
&lt;bb_height&gt;,
&lt;conf&gt;,
&lt;x&gt;,
&lt;y&gt;,
&lt;z&gt;
</code>

- frame: Die Nummer des Bildes (Frame)
- id: Objekt ID welche es zu bestimmen gilt. Deshalb ist diese im det.txt File immer -1
- bb_left: Linker Punkt der Bounding Box
- bb_top: Oberer Punkt der Bounding Box
- bb_width: Breite der Bounding Box
- bb_height: Höhe der Bounding Box
- conf: Zuverlässigkeit der Objekterkennung in Prozent
- x: Kann ignoriert werden
- y: Kann ignoriert werden
- z: Kann ignoriert werden

#### Beispiel
```
1,-1,246,218,40.258,91.355,69.358,-11.4773,-5.53043,0
1,-1,648,238,36.706,83.294,55.955,-8.82797,-12.7447,0
2,-1,495,158,30.979,70.299,98.284,-3.75501,-7.05616,0
2,-1,245,215,43.462,98.624,48.033,-11.6815,-5.68721,0
2,-1,637,245,33.73,76.541,33.712,-8.92948,-12.531,0
```
In dem Beispiel gibt es zwei Objekte in dem Frame 1 (`000001.jpg`) und drei Objekte im Frame 2 (`000002.jpg`)

- Frame 1:
  - Erstes Objekt: 
    - Konfidenz von: 69.358%
    - Position Bounding Box: 246,218,40.258,91.355
  - Zweites Objekt: 
    - Konfidenz von: 55.955%
    - Position Bounding Box: 648,238,36.706,83.294
- Frame 2:
  - Erstes Objekt: 
    - Konfidenz von: 98.284%
    - Position Bounding Box: 495,158,30.979,70.299
  - Zweites Objekt: 
    - Konfidenz von: 48.033%
    - Position Bounding Box: 245,215,43.462,98.624
  - Drittes Objekt: 
    - Konfidenz von: 33.712%
    - Position Bounding Box: 637,245,33.73,76.541

### gt.txt
Dieses File enthält die Ground Truth, dh.: die Detektionen (tatsächliche Position der Objekte) und eine ID 
(zeitlicher Zusammenhang).

#### Datenformat
Jede Zeile des Files enthält ein Objekt mit folgenden Werten:
<code>
&lt;frame&gt;,
&lt;id&gt;,
&lt;bb_left&gt;,
&lt;bb_top&gt;,
&lt;bb_width&gt;,
&lt;bb_height&gt;,
&lt;conf&gt;,
&lt;x&gt;,
&lt;y&gt;,
&lt;z&gt;
</code>

- frame: Die Nummer des Bildes (Frame)
- id: Die tatsächliche Objekt ID
- bb_left: Linker Punkt der Bounding Box
- bb_top: Oberer Punkt der Bounding Box
- bb_width: Breite der Bounding Box
- bb_height: Höhe der Bounding Box
- conf: Zuverlässigkeit der Objekterkennung in Prozent
- x: Kann ignoriert werden
- y: Kann ignoriert werden
- z: Kann ignoriert werden

### img Ordner
Hier befinden sich die Bilder passend zu den Framenummern in den Files gt.txt und det.txt

## Aufgaben
### 1. Daten einlesen und Bounding Boxen Zeichnen
Lesen sie mithilfe von Numpy die Detektionen ein und mithilfe von Pillow die dazugehörigen Bilder. Zeichnen Sie in die 
Bilder die Detektionen und speichern sie die entstandenen Bilder unter folgendem Pfad `dataset/out/` ab. Hier 
beispielsweise die Bilder für die ersten drei Frames.

Frame 1 | Frame 2 | Frame 3
--- | --- | --- 
<img src="img_2.png" alt="drawing" width="800px"/> | <img src="img_3.png" alt="drawing" width="800px"/> | <img src="img_4.png" alt="drawing" width="800px"/>

### 2. Funktion für das Überlappungsmaß
Das Überlappungsmaß, ist eine Metrik zur Bewertung der Genauigkeit von Bounding Boxen, insbesondere in Bezug auf 
Objekterkennung und -segmentierung. Sie wird häufig in Computer Vision-Anwendungen, wie der Objekterkennung und dem 
Tracking, verwendet. Das Überlappungsmaß kann folgendermaßen für zwei Bounding Boxen berechnet werden: 

Area_of_Overlap / Area_of_Union

<img src="img_5.png" alt="drawing" width="400px"/>

Das Überlappungsmaß soll mithilfe von Numpy vektorisiert berechnet werden. Die Methode sollte folgendermaßen definiert
werden.
```python
def ueberlappungsmass(box_1: np.array, box_2: np.array) -> float:
    # TODO
    return ueberlappungsmass

# mit box_1 [x1, y1, x2, y2] und box_2 [x1, y1, x2, y2]
res = ueberlappungsmass(np.array([495, 158, 525.979, 228.299]), np.array([500, 158, 530.979, 228.299]))
print(res)
0.7295762459774469
```

### 3. Berechnung und lösen der Kostenmatrix
Die Kostenmatrix enthält die Überlappungsmaße von allen Detektionen im Frame zu allen getrackten Objekten.

|       | Track 1 | Track 2 | Track 3 | Track 4
| --- |---------| --- | --- |  --- |
Detection 1 | 0.85 | 0.70 | 0.20 | 0.0
Detection 2 | 0.65 | 0.45 | 0.10 | 0.90
Detection 3 | 0.25 | 0.15 | 0.70 | 0.35

Diese Kostenmatrix kann mithilfe von `Hungarian Matching` gelöst werden. Verwenden Sie dazu folgende Funktion aus dem
`Scipy` Package.
```python
matched_det, matched_track = scipy.optimize.linear_sum_assignment(-iou_matrix)
```
Das Scipy Package kann folgendermaßen installiert werden.
```
pip install scipy
```

### 4. Matchen der Detektionen zu den Tracks
Matchen Sie mithilfe des Ergebnisses von `scipy.optimize.linear_sum_assignment` die Detektions zu den Tracks und 
Zeichnen Sie diese in die dazugehörigen Frames ein. Für eine bessere Visualisierung nutzen Sie unterschiedliche Farben
für unterschiedliche IDs. Dies kann mit einer endlichen Liste von Farben gemacht werden. Z.B.: Eine Liste mit 12 Farben
die sich nach jeder 12 ID wiederholt. Weiteres soll über die Bounding Box die ID eingezeichnet werden.

Frame 1 | Frame 2 | Frame 3
--- | --- | --- 
![img_6.png](img_6.png) | ![img_7.png](img_7.png) | ![img_8.png](img_8.png)!

Schreiben anschließend die Ergebnisse in ein File in dem Pfad `dataset/res/res.txt` mit folgendem Datenformat:
<code>
&lt;frame&gt;,
&lt;id&gt;,
&lt;bb_left&gt;,
&lt;bb_top&gt;,
&lt;bb_width&gt;,
&lt;bb_height&gt;,
&lt;-1&gt;,
&lt;-1&gt;,
&lt;-1&gt;,
&lt;-1&gt;
</code>

Beispiel für die ersten drei Objekte im ersten Frame:

`1, 1, 500.0, 158.0, 30.979000000000042, 70.299, -1, -1, -1, -1
1, 2, 246.0, 218.0, 40.25799999999998, 91.35500000000002, -1, -1, -1, -1
1, 3, 648.0, 238.0, 36.70600000000002, 83.29399999999998, -1, -1, -1, -1`

### 5. Evaluieren der Ergebnisse
Nutzen sie das Script `eval.py` um Ihre Ergebnisse im File `dataset/res/res.txt` gegen die GT `dataset/gt/gt.txt` zu
evaluieren. 

Sie sollten beispielsweise eine Ausgabe der folgenden Form erhalten:

| num_frames | idf1 | idp | idr | recall | precision | num_objects | mostly_tracked | partially_tracked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 795 | 0.289712 | 0.25986 | 0.327312 | 0.905806 | 0.719139 | 4650 | 18 | 1 |

| mostly_lost | num_false_positives | num_misses | num_switches | num_fragmentations | mota | motp |
|-------------| --- | --- | --- | --- | --- | --- |
| 0 | 1645 | 438 | 225 | 147 | 0.503656 | 0.292089 |

Des Weiteren wird ein Outputfile `dataset/res/out_{timestamp}.txt` geschrieben wo Sie ihre Ergebnisse vergleichen können. Um das 
Script `eval.py` verwenden zu können muss folgendes Package installiert werden:
```
pip install motmetrics
```

### 6. Tipps
1. Erstellen Sie eine Klasse TrackerBox mit folgenden Klassenvariablen
   - vier Eckpunkten
   - der ID 
   - und einer Variable, welche hochgezählt wird wenn die Box in einem Frame nicht gematched wurde. Die kann dazu verwendet werden, um die Box nach der Zeit wieder zu löschen.
2. Eine Liste mit Tracks welche TrackerBoxen enthält die mit den Detektions gematched werden. Damit diese Liste nicht unendlich lang wird, sollten alte Trackerboxen welche nach einiger Zeit nicht gemachted wurden wieder gelöscht werden.
3. Sauber arbeiten!

