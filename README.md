# SIIM-ISIC Melanoma Classification SoSe24
https://www.kaggle.com/competitions/siim-isic-melanoma-classification

# 1. Einleitung
Eine primäre Datenquelle für die Entwicklung unseres Fallbeispiels ist der SIIM-ISIC
Melanoma Classification Datensatz, der auf der Plattform Kaggle verfügbar ist. Dieser
Datensatz enthält eine Vielzahl von Bildern und Metadaten, die speziell für die Aufgabe der
Melanomerkennung zusammengestellt wurden. Der Kaggle-Datensatz umfasst:
- Bilddaten
  - Hochauflösende Bilder von Hautläsionen in verschiedenen Formaten. Diese
Bilder sind notwendig, um das Modell auf die visuelle Erkennung von
Melanomen zu trainieren.
- Metadaten
  - Begleitende CSV-Dateien, die zusätzliche Informationen zu den Bildern
enthalten, wie z.B. das Geschlecht, das Alter des Patienten, die anatomische
Stelle der Läsion und die Diagnose (Melanom oder gutartig).
Bildformate und -typen
Die Bilder in dem Kaggle-Datensatz liegen in verschiedenen Formaten vor, die jeweils
unterschiedliche Eigenschaften und Nutzungsmöglichkeiten bieten:
- JPEG
  - Dieses weit verbreitete Bildformat bietet eine gute Balance zwischen
Bildqualität und Dateigröße, was es ideal für die Verarbeitung großer
Bildmengen macht. JPEG-Bilder sind einfach zu handhaben und kompatibel
mit den meisten Bildverarbeitungsbibliotheken.
- DICOM
  - Das DICOM-Format (Digital Imaging and Communications in Medicine) wird
häufig in der medizinischen Bildgebung verwendet. Es enthält nicht nur das
Bild selbst, sondern auch umfangreiche Metadaten, die wichtige
Kontextinformationen für die medizinische Diagnose liefern können.
Verwendung von JPEG- und CSV-Daten
Für die Entwicklung unseres KI-gestützten Systems zur Identifikation von Melanomen haben
wir uns entschieden, ausschließlich JPEG-Bilddaten und begleitende CSV-Metadaten,
speziell dazu nur die Target-Labels, zu verwenden.

# 2. Identifizierung des KI-Anwendungsfalls

## 2.1 Welches Problem oder welche Aufgabe soll gelöst werden?
**Identifikation von Melanomen in Hautläsionen** 

Die Früherkennung von Hautkrebs, insbesondere von Melanomen, stellt eine bedeutende
medizinische Herausforderung dar. Melanome sind die gefährlichste Form von Hautkrebs und
können, wenn sie nicht frühzeitig erkannt und behandelt werden, schnell metastasieren und
lebensbedrohlich werden. Die rechtzeitige Diagnose und Behandlung von Melanomen können
die Überlebensraten erheblich verbessern und die Morbidität und Mortalität reduzieren. (vgl.
Better health Channel, 2024)

**Unterstützung und Entlastung von Dermatologen**

In der klinischen Praxis stehen Dermatologen vor der anspruchsvollen Aufgabe, eine Vielzahl
von Hautläsionen zu bewerten, von denen einige bösartig sein können. Diese Aufgabe ist nicht
nur zeitaufwendig, sondern erfordert auch ein hohes Maß an Fachwissen und Erfahrung.
Angesichts der wachsenden Anzahl von Patienten, die eine dermatologische Untersuchung
benötigen, und des begrenzten Fachpersonals im Gesundheitswesen, besteht ein dringender
Bedarf an Unterstützung durch innovative Technologien. (vgl. Heidelberg, 2024)

**Früherkennung von Hautkrebs**

Die Früherkennung von Melanomen ist entscheidend, da die Heilungschancen in frühen
Stadien deutlich höher sind. Traditionelle Diagnosemethoden, die hauptsächlich auf visuellen
Inspektionen und Dermatoskopie basieren, sind subjektiv und können zu variierenden
Ergebnissen führen. Daher ist die Entwicklung objektiver und automatisierter Systeme zur
Erkennung von Melanomen von großer Bedeutung. (vgl. Dr. P. Mohr, Dr. med. S. Lutze, A.
Arnold, 2023)

**Reduzierung von Todesfällen**

Durch die frühzeitige Erkennung von Melanomen kann die Behandlung früher eingeleitet
werden, was die Heilungschancen erhöht und die Notwendigkeit invasiver Eingriffe verringert.
Dies führt zu einer insgesamt besseren Prognose für die Patienten und kann die Todesrate, die
mit fortgeschrittenem Hautkrebs verbunden ist, erheblich reduzieren. (vgl. Dr. P. Mohr, Dr. med.
S. Lutze, A. Arnold, 2023)

**Verbesserung der Diagnosegenauigkeit**

Ein zentrales Ziel der KI-Entwicklung ist die Verbesserung der Diagnosegenauigkeit im
Vergleich zu herkömmlichen Methoden. Durch den Einsatz von maschinellem Lernen und
Bildverarbeitungstechniken soll die KI in der Lage sein, Melanome mit hoher Präzision zu
erkennen und zu klassifizieren, wodurch die Anzahl der falsch-positiven und falsch-negativen
Diagnosen reduziert wird. (vgl. Better health Channel, 2024)

**Erhöhung der Effizienz und Entlastung des Fachpersonals**

Ein weiterer wesentlicher Aspekt ist die Erhöhung der Effizienz in der dermatologischen Praxis.
Durch die Automatisierung der Erkennung und Klassifizierung von Hautläsionen können
Dermatologen entlastet werden und sich auf komplexere Fälle und andere klinische Aufgaben
konzentrieren. Dies führt zu einer besseren Ressourcennutzung im Gesundheitswesen und
einer schnelleren Patientenversorgung. ( vgl. Frederik Wenz & Stefan Ebener, 2024, S.6 )

## 2.2 Welche Geschäftsziele sind zu erreichen ?

**Erhöhung der Produktivität**

Ein zentrales Ziel der Einführung eines KI-gestützten Diagnosewerkzeugs ist die Erhöhung
der Produktivität in dermatologischen Praxen und Kliniken. Die Automatisierung der ersten
Analyse von Hautläsionen ermöglicht es Dermatologen, ihre Zeit und Aufmerksamkeit auf
komplexere und schwerer zu diagnostizierende Fälle zu konzentrieren. Dies führt zu einer
effizienteren Nutzung der Fachkompetenz und einer verbesserten Patientenversorgung.
Studien haben gezeigt, dass die Einbindung von KI in medizinische Arbeitsprozesse zu
erheblichen Zeiteinsparungen und einer erhöhten Produktivität führen kann.(vgl. Frederik
Wenz & Stefan Ebener, 2024, S.603)

**Unterstützung der Entscheidungsprozesse und Effizienzsteigerung**

KI-Systeme können als Entscheidungsunterstützung dienen, indem sie genaue und
konsistente Analysen von Hautläsionen liefern. Diese Unterstützung kann dazu beitragen,
die Entscheidungsprozesse in der medizinischen Diagnostik zu verbessern und Fehler zu
reduzieren. Durch die Bereitstellung zusätzlicher Informationen und Empfehlungen auf Basis
umfangreicher Datenanalysen kann die KI den Ärzten helfen, fundierte Entscheidungen zu
treffen. Dies führt nicht nur zu einer Effizienzsteigerung, sondern auch zu einer höheren
Qualität der Diagnosen und Behandlungen. Die Vermeidung von Fehldiagnosen und die
Reduzierung von unnötigen Biopsien sind konkrete Vorteile, die durch den Einsatz von KI
erzielt werden können. (vgl. Frederik Wenz & Stefan Ebener, 2024, S.6)

**Risikominderung durch frühzeitige Erkennung**

Ein weiteres wesentliches Geschäftsziel ist die Risikominderung durch die frühzeitige
Erkennung von Melanomen. Durch die Implementierung eines zuverlässigen KI-Systems
können Melanome in einem frühen Stadium identifiziert werden, was die Chancen auf eine
erfolgreiche Behandlung erheblich erhöht. Die frühzeitige Erkennung und Behandlung von
Hautkrebs kann die Morbidität und Mortalität senken und somit die allgemeine
Gesundheitsprognose verbessern. Die Möglichkeit, schnell und präzise auf verdächtige
Hautveränderungen zu reagieren, trägt dazu bei, das Risiko für Patienten zu minimieren und
die Gesundheitsausgaben langfristig zu senken. (vgl. Frederik Wenz & Stefan Ebener, 2024,
S.178)

## 2.3 Anhand welcher Kennzahlen kann die Zielerreichung bewertet werden ?

Die erfolgreiche Implementierung und der Nutzen eines KI-gestützten Systems zur
Identifikation von Melanomen können durch eine Reihe spezifischer Kennzahlen bewertet
werden. Diese Kennzahlen sind entscheidend, um die Leistungsfähigkeit des Modells zu
beurteilen, potenzielle Verbesserungsbereiche zu identifizieren und sicherzustellen, dass die
Geschäftsziele erreicht werden. Im Folgenden werden die wichtigsten Kennzahlen erläutert,
die zur Bewertung der Zielerreichung herangezogen werden.

**Genauigkeit (Accuracy)**

Die Genauigkeit ist eine grundlegende Kennzahl zur Bewertung der Leistungsfähigkeit eines
Klassifikationsmodells. Sie gibt den Anteil aus, wie oft das Modell richtig liegt . Eine hohe
Genauigkeit ist ein Indikator dafür, dass das Modell sowohl Melanome als auch gutartige
Hautläsionen korrekt identifiziert. Diese Kennzahl ist jedoch allein nicht ausreichend,
insbesondere bei einem unausgewogenen Datensatz, bei dem die Anzahl der Melanome im
Vergleich zu gutartigen Fällen gering ist.

**Anteil falsch-positiver und falsch-negativer Ergebnisse**

Der Anteil der falsch-positiven (False Positives) und falsch-negativen (False Negatives)
Ergebnisse ist eine kritische Kennzahl. Falsch-positive Ergebnisse führen zu unnötigen
Behandlungen und Belastungen für die Patienten, während falsch-negative Ergebnisse
gefährlich sind, da sie zu einer verspäteten Diagnose und Behandlung von Melanomen
führen können.

**Welche Daten und Informationen liegen vor und können beschafft werden?**

Die Entwicklung eines KI-gestützten Systems zur Identifikation von Melanomen in
Hautläsionen erfordert den Zugriff auf umfangreiche und qualitativ hochwertige Daten. Diese
Daten sind essenziell, um das Modell zu trainieren, zu validieren und letztlich in der
klinischen Praxis erfolgreich einzusetzen. In diesem Fall haben wir Zugang zu SIIM-ISIC
Melanoma Classification in Kaggle, welches etwa 33 tausend Trainingsbilder und 10 tausend
Testbilder uns zur Verfügung stellt sowie die dazugehörigen Metadaten.

# 3. Potenzialanalyse der KI
## 3.1 Welchen Beitrag kann ein KI-System zur Lösung des Problems leisten?

**Verbesserung der Diagnosen**

Ein KI-gestütztes System kann die Genauigkeit und Konsistenz von Diagnosen erheblich
verbessern. Durch den Einsatz fortschrittlicher Algorithmen und tiefen Lernens (Deep Learning)
ist die KI in der Lage, große Mengen an Bilddaten zu analysieren und Muster zu erkennen, die
für das Menschliche Auge schwer zu erkennen sind. Dies führt zu einer präziseren Diagnosen
und Hautläsionen und einer besseren Unterscheidung zwischen gutartigen und bösartigen
Veränderungen. Die höhere Genauigkeit bei der Melanomerkennung kann dazu beitragen, die
Zahl der Fehldiagnosen zu reduzieren und somit die Patientenversorgung zu verbessern(vgl.
Ludovic Amruthalingam, 2024)

**Schnellere Diagnosen**

Ein wesentlicher Vorteil eines KI-Systems ist die Geschwindigkeit, mit der Diagnosen gestellt
werden können. Während ein Dermatologe Zeit benötigt, um jeden einzelnen Fall zu
überprüfen und zu bewerten, kann ein KI-System innerhalb von Sekunden Tausende von
Bildern analysieren. Dies führt zu einer erheblichen Zeitersparnis und ermöglicht es den
Ärzten, mehr Patienten in kürzerer Zeit zu behandeln. Die beschleunigte Diagnosestellung
ist besonders in überlasteten Gesundheitssystemen von großem Nutzen.(vgl. Frederik Wenz
& Stefan Ebener, 2024, S.179)

## 3.2 Welche Aufgaben hat der Mensch weiterhin, ergänzend zum KI-System?

**Ergänzende Aufgaben des Menschen zum KI-System**

Obwohl KI-Systeme in der Lage sind, eine Vielzahl von Aufgaben effizient und präzise zu
übernehmen, bleibt die Rolle des Menschen im Zusammenspiel mit diesen Technologien
weiterhin entscheidend. Ein KI-gestütztes System zur Identifikation von Melanomen kann die
Arbeit von Dermatologen erheblich erleichtern und verbessern, doch es erfordert weiterhin
menschliche Expertise und Intervention in mehreren Schlüsselbereichen. Im Folgenden
werden die wesentlichen Aufgaben des Menschen beschrieben, die ergänzend zum
KI-System notwendig sind.

**Wartung und Management**

Die Implementierung eines KI-Systems erfordert eine kontinuierliche Wartung und
Verwaltung. Dies umfasst die Überwachung der Systemleistung, die Aktualisierung der
Software und die Sicherstellung, dass die Hardware den Anforderungen entspricht.
Technische Probleme müssen identifiziert und behoben werden, um einen reibungslosen
Betrieb des Systems zu gewährleisten. (vgl. Frederik Wenz & Stefan Ebener, 2024, S.279)

**Stetige Anpassung und Aktualisierung**

KI-Systeme müssen regelmäßig aktualisiert und angepasst werden, um mit den neuesten
medizinischen Erkenntnissen und technologischen Entwicklungen Schritt zu halten. Dies
erfordert die Anpassung der Algorithmen und Modelle sowie die Integration neuer Daten.
Menschen spielen eine entscheidende Rolle bei der Identifikation von
Verbesserungsmöglichkeiten und der Implementierung notwendiger Updates, um die
Effizienz und Genauigkeit des Systems zu erhalten.

**Überwachung und Kontrolle der KI-Entscheidungen**

Menschen müssen die Entscheidungen und Vorschläge der KI überwachen und
kontrollieren. Dies ist besonders wichtig, um sicherzustellen, dass die KI-Entscheidungen mit
den ethischen Standards und medizinischen Richtlinien übereinstimmen. Eine kontinuierliche
Überwachung ermöglicht es, potenzielle Fehler oder Missverständnisse zu identifizieren und
zu korrigieren, bevor sie negative Auswirkungen auf die Patientenversorgung haben. (vgl.
Susanne Beck, Michelle Faber & Simon Gerndt LL. B., 2023, S.260)

## 3.3 Wer profitiert von der Implementierung des KI-Systems?

**Nutzen und Profiteure der Implementierung eines KI-Systems**

Die Implementierung eines KI-gestützten Systems zur Identifikation von Melanomen bringt
vielfältige Vorteile mit sich und betrifft eine Vielzahl von Stakeholdern. Im Folgenden werden
die Hauptprofiteure beschrieben und die spezifischen Vorteile erläutert, die sie durch den
Einsatz des KI-Systems erfahren.

**Patienten**
Die Patienten sind die Hauptnutznießer der Implementierung eines KI-Systems in der
dermatologischen Diagnostik. Zu den wichtigsten Vorteilen für Patienten gehören:
- Schnellere Diagnosen: Die automatisierte Analyse von Hautläsionen ermöglicht
eine schnellere Diagnosestellung, wodurch Patienten schneller Klarheit über ihren
Gesundheitszustand erhalten. Dies kann insbesondere in Fällen von Hautkrebs
lebensrettend sein, da die Früherkennung entscheidend für die Behandlungserfolge
ist. (vgl. inovex, 2024)
- Präzisere Diagnosen: Die hohe Genauigkeit und Konsistenz des KI-Systems
reduzieren die Wahrscheinlichkeit von Fehldiagnosen. Patienten profitieren von
zuverlässigeren Ergebnissen und können sicherer sein, dass ihre Diagnose korrekt
ist. (vgl. inovex, 2024)
- Vermeidung von Leid: Eine frühzeitige und präzise Diagnose kann das Leiden der
Patienten verringern, indem unnötige und invasive Untersuchungen vermieden
werden. Zudem kann eine rechtzeitige Behandlung eingeleitet werden, was die
Prognose und die Lebensqualität der Patienten verbessert. (vgl. inovex, 2024)

**Ärzte**

Auch Ärzte und medizinisches Fachpersonal profitieren erheblich von der Implementierung
eines KI-Systems. Zu den Vorteilen gehören:
- Entlastung der Arbeit: Die Automatisierung der Hautläsionenanalyse entlastet
Dermatologen und ermöglicht es ihnen, ihre Zeit und Energie auf komplexere und
dringendere Fälle zu konzentrieren. Dies erhöht die Effizienz und Produktivität der
medizinischen Fachkräfte. (vgl. inovex, 2024)
- Effizienterer Ressourceneinsatz: Durch die Unterstützung des KI-Systems können
Ärzte ihre Ressourcen besser nutzen. Dies führt zu einer effizienteren Terminplanung
und einer besseren Nutzung der verfügbaren diagnostischen Werkzeuge. (vgl.
inovex, 2024)
- Unterstützung bei der Entscheidungsfindung: KI-Systeme bieten zusätzliche
Informationen und Analysen, die Ärzte in ihrer Entscheidungsfindung unterstützen.
Dies führt zu fundierteren Entscheidungen und einer höheren Qualität der
medizinischen Versorgung. (vgl. inovex, 2024)
Krankenkassen
Krankenkassen können ebenfalls von der Implementierung eines KI-Systems profitieren,
insbesondere durch die Reduzierung der Gesundheitskosten:
- Einsparung von Behandlungskosten: Durch die frühzeitige Erkennung von
Melanomen und die Vermeidung unnötiger Behandlungen können die
Behandlungskosten erheblich gesenkt werden. Früh erkannter Hautkrebs erfordert oft
weniger intensive und kostspielige Behandlungen als fortgeschrittener Krebs. (vgl.
inovex, 2024)
- Effiziente Ressourcennutzung: Die effizientere Nutzung der medizinischen
Ressourcen durch den Einsatz von KI-Systemen kann zu einer insgesamt
kostengünstigeren Gesundheitsversorgung führen. Dies kann sich positiv auf die
Prämien und die finanzielle Stabilität der Krankenkassen auswirken. (vgl. inovex,
2024)
  
## 3.4 Wie gestalten wir Transparenz und Erklärbarkeit intern?
Ein effektives Mittel, um die Erklärbarkeit von KI-Systemen im Bereich der
Melanomklassifikation zu erhöhen, ist die Visualisierung der Entscheidungsprozesse durch
Heatmaps. Diese grafischen Darstellungen bieten wertvolle Einblicke in die
Entscheidungsfindung der KI, insbesondere bei der Diagnose von Hautkrebs.

**Funktion von Heatmaps in der Melanoma Classification**

Heatmaps visualisieren, welche Bereiche eines Hautbildes von der KI besonders stark
berücksichtigt wurden, um eine Entscheidung über das Vorhandensein eines Melanoms zu
treffen. Dies erfolgt durch farbliche Markierungen, die die Wichtigkeit verschiedener
Bildbereiche anzeigen.

**Evaluierung und Validierung**

Die Evaluierung und Validierung eines KI-Systems sind entscheidend, um seine
Zuverlässigkeit und Genauigkeit zu gewährleisten. Dies erfolgt durch:
Prüfung der KI mit eigenen und Drittdaten: Um sicherzustellen, dass das System unter
verschiedenen Bedingungen zuverlässig funktioniert, wird es mit verschiedenen Datensätzen
getestet.
Kontinuierliche Überwachung der Leistung: Regelmäßige Überprüfungen und
Aktualisierungen der KI-Modelle helfen, deren Leistung aufrechtzuerhalten und zu
verbessern.

**Detaillierte Dokumentation**

Eine ausführliche Dokumentation aller Prozesse und Entscheidungen der KI ist notwendig,
um ihre Nachvollziehbarkeit zu gewährleisten. Diese sollte umfassen:
Erklärung der Modellarchitektur und der verwendeten Algorithmen: Beschreibungen, wie das
neuronale Netz aufgebaut ist und welche Algorithmen verwendet werden.
Protokollierung der Trainings- und Testdaten: Informationen über die Herkunft und die
Eigenschaften der Daten, die für das Training und die Evaluierung der KI verwendet werden.

**Transparente Datennutzung**

Die Herkunft und Qualität der verwendeten Daten müssen offen kommuniziert werden. Dies
beinhaltet:
Informationen über Datenquellen: Offenlegung, woher die Daten stammen und unter
welchen Bedingungen sie gesammelt wurden.
Qualitätskontrollen: Beschreibungen der Maßnahmen zur Sicherstellung der Datenqualität,
wie z.B. Filterung und Bereinigung der Daten.

# 4. Vorgehensweise

Im Folgenden erläutern wir unsere Vorgehensweise zu dieser Fallstudie. Ursprünglich haben wir
versucht, mit tfrecords-Dateien zu arbeiten, sind jedoch zu keinem Ergebnis gekommen, da wir
nicht verstehen konnten, wie diese Dateien geladen und verwendet werden. Der Prozess wurde
am Beispiel zur Handschrifterkennung leider nicht erklärt. Daher haben wir uns letztendlich für
die Verwendung von JPEG-Dateien entschieden.

## 4.1 Libraries importieren

Dieser Block importiert essentielle Bibliotheken wie NumPy, Pandas, TensorFlow, Keras und
die Python Imaging Library (PIL). Diese Bibliotheken sind zum Einlesen, Anzeigen,
Bearbeiten und Trainieren der Daten notwendig. NumPy ermöglicht unter anderem die
Weitergabe der Datensätze in Form von Arrays an TensorFlow.keras und somit die weitere
Verarbeitung und das Training des Modells. Pandas ermöglicht das Einlesen, Erstellen,
Verarbeiten und Analysieren von Daten, in unserem Fall .csv Daten.PIL wird in unserem Fall
nur zum Laden der Bilder verwendet.
Besondere Bibliotheken sind hier ThreadPoolExecutor, tqdm, Matplotlib und Scikit-learn.
ThreadPoolExecutor ermöglicht die parallele Ausführung von Funktionen mithilfe von Threads
oder Teilprozessen. Tqdm ermöglicht das Anzeigen von Ladebalkens. Matplotlib wird zur
Visualisierung unserer Ergebnisse verwendet. Scikit-learn wird zum Aufteilen unseres
Trainingdatensatzes in Trainings- und Validierungssätze sowie zur Generierung von
Ergebnisvisualisierungen verwendet.

![grafik](https://github.com/user-attachments/assets/514288eb-dda8-46a9-8428-5e4b2dac365c)

<p style="text-align: center;"> Abbildung 1: Import Libraries </p>

## 4.2 Daten laden

Beginnend zum Laden der relevanten Daten werden die Pfade zu den Bildern und .csv Dateien
bestimmt.
![grafik](https://github.com/user-attachments/assets/e2797af1-190d-42ad-93da-2507fe60094e)

<p style="text-align: center;">Abbildung 2: Pfade definieren </p>

Ferner werden Funktionen zum Laden der Bilder sowie der Bildnamen und Ziel-Labels definiert.
Die Bilder werden in einer Auflösung von 128x128 Pixeln mit drei Farbkanälen in den RAM
geladen, zusammen mit den Bildnamen und Labels. Um die Ladezeiten zu verkürzen, werden
die Bilder multithreaded geladen. Dies reduziert die Ladezeiten von etwa drei Stunden auf eine
Stunde auf Kaggle bzw. von etwa einer Stunde auf zehn Minuten auf dem Heim-PC.
Wir haben die Auflösung von 256x256 auf 128x128 reduziert, da dies anstelle von etwa 25GB
RAM nur noch etwa 12 GB RAM benötigt und somit auch in den VRAM der GPU geladen
werden kann. Darüber hinaus führt eine Überlastung von RAM oder VRAM auf Kaggle zum
Neustart der Umgebung, was das erneute Laden des Datensatzes bedeutet.
![grafik](https://github.com/user-attachments/assets/d7159813-9ea7-4651-9a55-a449f861a1d5)

<p style="text-align: center;"> Abbildung 3: Funktion zum Laden der Bilder und Labels </p>

In der folgenden Abbildung werden die Ladefunktionen aufgerufen und die Trainingsdaten in
Trainings- und Validierungssätze aufgeteilt.
![grafik](https://github.com/user-attachments/assets/f45e07aa-5376-4e44-911a-e52a1202a71d)

<p style="text-align: center;"> Abbildung 4: Aufruf der Ladefunktionen </p>
## 4.3 Daten augmentieren
Die folgende Abbildung zeigt den Datenaugmentierungsblock. Hierbei haben wir uns
überlegt, die Verteilung von ~98% gutartig und ~2% bösartig auf 50-50% zu augmentieren,
um zu vermeiden, dass das Modell ausschließlich “gutartig” ausgibt und damit eine
Genauigkeit von ~98% erreicht.
Die augment_class_images Funktion nimmt, Bilder, Augmentierungsgröße und Parameter
der ImageDataGenerator und augmentiert bösartig markierte Bilder um etwa das 56-Fache
auf eine Verteilung von 50%.
![grafik](https://github.com/user-attachments/assets/8ca4a4a8-17e5-43b1-8d9b-9269350fc5ef)

Abbildung 5: Datenaugmentierung
Nach der Augmentierung werden die augmentierten Datensätze mit ihren jeweiligen
ursprünglichen Datensätzen kombiniert und schließlich gemischt.
![grafik](https://github.com/user-attachments/assets/1af2b903-0a9c-40bd-b742-0d7eae1832cc)

Abbildung 6: Mischen der augmentierten Datensätze
## 4.4 Künstliches Neuronales Netz entwerfen und konfigurieren
Der folgende Block definiert ein CNN (Convolutional Neural Network) mit einem Input von
128x128 und drei Farbkanälen und setzt sich aus jeweils drei Convolutional2D und
MaxPooling2D Schichten zusammen.
Die Conv2D Schichten starten bei 32 und verdoppeln sich pro Ebene bis 256. Die Conv2D
Schichten dienen der Merkmalsextraktion1.
MaxPooling2D Schichten werden zwischen die Conv2D Schichten geschaltet und halbieren
die räumliche Größe der Matrix auf die wichtigsten Pixel2.
Flatten transformiert die drei Dimensionale Matrix in eine ein Dimensionale Matrix, um diese
an die folgende Dense übergeben zu können.
Dropout deaktiviert hier 30% zufälliger Neuronen, um Overfitting zu vermeiden.

![grafik](https://github.com/user-attachments/assets/bd6d4512-3b51-4f19-a1b0-5c7b06c01f00)

Abbildung 7: Entwurf des Künstlichen Neuronalen Netzes

## 4.5 Modell kompilieren und trainieren

Das Modell wird mit dem Optimizer "Adam" und einer Lernrate von 0,0001 sowie einer binären
Kreuzentropie-Verlustfunktion kompiliert. Das Early Stopping überwacht die val_loss Metrik und
stoppt das Training, wenn diese sich über zwei Epochen nicht verbessert hat. Es stellt das
Modell dann auf den Zustand der letzten Epoche mit den besten Werten zurück. Das Modell
wird mit einer Batchgröße von 32 und maximal 100 Epochen trainiert.
![grafik](https://github.com/user-attachments/assets/5cc0ad11-daca-41f2-9a17-fdc437537d0d)

Abbildung 8: Modell kompilieren und trainieren; Early Stopping

## 4.6 Visualisierung der Ergebnisse
Die Verluste, Genauigkeiten und Konfusionsmatrix werden in den folgenden zwei Blöcken
erstellt und generiert. Die Ergebnisse der Visualisierungen sind in Kapitel 6 vorzufinden.
![grafik](https://github.com/user-attachments/assets/b003371c-f0c0-4510-af25-2605632af3f4)

Abbildung 9: Erstelle Graphen zu Verlusten und Genauigkeiten

![grafik](https://github.com/user-attachments/assets/d2e50ee0-4dd6-4fe0-af8b-f55ad7093996)

Abbildung 10: Erstelle Konfusionsmatrix

# 5. Lern- und Validierungsergebnisse

Im Folgenden werden die Lern- und Validierungsergebnisse dargestellt. Diese Ergebnisse
basieren auf einer Trainingsdauer von 100 Epochen und zeigen die Verlust- bzw.
Genauigkeitswerte auf der Y-Achse. Obwohl das Early Stopping bei zwei
aufeinanderfolgenden schlechter werdenden val_loss-Werten eingestellt ist, konnte das
Modell bis zur maximalen Epochenzahl trainiert werden.
![grafik](https://github.com/user-attachments/assets/4680683c-c6c3-43ee-ae1d-5d601a5e9378)

Abbildung 11: Output des Trainingsprozesses

## 5.1 Trainings- und Validierungsgenauigkeiten
Die folgende Grafik zeigt die Trainings- und Validierungsgenauigkeiten auf 100 Epochen an.
Es fällt auf, dass die Trainingsgenauigkeit sich den 100% annähert und die
Validierungsgenauigkeiten auf etwa 95% sich einpendelt.
![grafik](https://github.com/user-attachments/assets/adcd0256-4cdc-4d12-90e6-2e9d504cbc9f)

Grafik 12: Trainings- und Validierungsgenauigkeiten
## 5.2 Trainings- und Validierungsverluste
Die folgende Grafik zeigt die Trainings- und Validierungsverluste auf 100 Epochen an. Es
fällt auf, dass die Validierungsverluste in den ersten etwa 10 Epochen abnehmen und
danach einen steigenden Trend aufweisen, während die Trainingsverluste einen stetig
sinkenden Trend aufweisen.
![grafik](https://github.com/user-attachments/assets/dc194858-7790-41ef-aa41-6df395221a20)

Grafik 13: Trainings- und Validierungsverluste

## 5.3 Konfusionsmatrix
In der folgenden Grafik ist eine Konfusionsmatrix zu sehen. An der Y-Achse sind die
Vorhersagen des Modells von Positiv zu Negativ und an der X-Achse die wahren Werte von
Negativ zu Positiv dargestellt. Es ist zu erkennen, dass besonders hohe Werte in den “True
Negatives” und “True Positives” zu finden sind.
![grafik](https://github.com/user-attachments/assets/89d73c78-d46c-4f38-97c0-752a4e69ea32)

Grafik 3: Konfusionsmatrix


# Fazit

Zu Beginn unserer Bearbeitung waren wir hoch motiviert, eine Melanomerkennungs-KI zu
entwickeln, da eine Person bereits Python Kenntnisse vorweisen konnte. Jedoch sind wir
früh auf ein Problem gestoßen, nämlich dass das Laden der Beispieldatei zur
Handschrifterkennung nicht tiefer behandelt wurde. Daher sind wir auf Beispiele im Internet
gestoßen, wie Datensätze vorab in den Arbeitsspeicher(RAM) geladen werden. Wir sind uns
sicher, dass diese Methode viele Probleme mit sich gezogen hat, unter anderem, dass der
RAM bzw. VRAM überlastet werden und die Umgebung regelmäßig abstürzen lässt. Eine
Alternative, die wir dazu gefunden hatten, war, dass die Bilder während des
Trainingsprozesses geladen wurden. Dies führte jedoch dazu, dass jede Epoche übermäßig
lange (über 3h) benötigt hat. Außerdem hat man mit unserer Methode die Möglichkeit zu
multithreaden (von 3h auf 1h mit ~33.000 Trainingsbildern).
Auf weitere Probleme sind wir bei der Validierung unserer Trainingsdaten gestoßen, da
unser Modell zu jedem Validierungsbild denselben Wert ausgegeben hatte. Das Problem ließ
sich mit der Augmentierung und Einfügen einer Dropout-Schicht beheben. Dadurch scheint
das Modell zu funktionieren.
Während des Prozesses sind wir auf das Problem gestoßen, dass die Metadaten nicht
vollständig ausgefüllt waren, beispielsweise, dass 82% der Diagnosen unbekannt waren und
nur die Targets vollständig waren.
Schlussendlich hat das Projekt Spaß gemacht, besonders da man sich regelmäßig mit
neuen Herausforderungen auseinandersetzen musste. Wir hatten dennoch große Probleme,
welche Best-Practices angewandt werden müssten.

# Zusatz

Aus Interesse haben wir unsere
Ausarbeitung zur Bewertung in
Kaggle eingeschickt und sind
schlussendlich zu einem Score von
0,6688 angelangt.
![grafik](https://github.com/user-attachments/assets/2742776a-865b-411b-9d34-220e4be4bcd3)

--------------------------------------------------------------------------------------------------------------------------
This is a student project I did for university.
This model has an accuracy of about 60% in Kaggle and uses the JPEG Dataset provided by Kaggle and a Convolutional Neural Network, as I couldnt figure out how to work with tfrecords or dicom. On later iterations I would have used resnet or effnet but I wanted to design my own Neural Network.

If you have to do something similar feel free to use my code.

PS: You need to use python 3.9.19 (on Windows) if you want to use your GPU.


