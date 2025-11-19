# Examination Scheduling Optimization using N-Queens Problem  
**BITE4077L – Project**  
**Course:** Quantum Computing  
**Faculty Guide:** Aswani Kumar Cherukuri  
**Institution:** School of Computer Science Engineering and Information Systems, VIT Vellore  

## Project Members  
- **Keshav Malik (22BIT0026)**  
- **Harsh Gulati (22BIT0068)**  

---

## Description  
This project implements **exam scheduling optimization** using the classical **N-Queens problem** as an analogy. Each examination is modeled as a queen placed on an **N×N grid**, where:

- **Rows → Time slots (0..N-1)**  
- **Columns → Rooms (0..N-1)**  
- **Queens → Exams placed at (time_slot, room)**  

### **Scheduling Constraints**  
- Only **one exam per time slot** (one queen per row).  
- Only **one exam per room** (one queen per column).  
- Exams sharing students **must not occur at the same time** (diagonal-like conflict constraint).  

This satisfies the N-Queens formulation and provides a structured system for organizing conflict-free exam schedules.

---

## Features Implemented  
### **1. Classical Solvers**
- **Backtracking Solver (Exact):**  
  Generates valid schedules by exploring placements systematically.  
- **Min-Conflicts Solver (Heuristic):**  
  A fast, local-search solver suitable for larger N.

### **2. Quantum Component (Course Requirement)**  
A demonstration of **Grover’s Search Algorithm** using Qiskit:

- Identifies a marked state (e.g., `|101⟩`)  
- Uses **superposition + amplitude amplification**  
- Shows a **visible histogram** using the AerSimulator  

This demonstrates how quantum search can accelerate retrieval from a large set of valid schedules.

### **3. Visualization Tools**  
- Schedule plotter to show timeslot-room allocation on an N×N grid.  
- Helps verify correctness of solutions visually.

---

## Code Structure  
| File / Function          | Description |
|--------------------------|-------------|
| `schedule_is_valid()`    | Validates placements + conflict pairs |
| `solve_backtracking()`   | Exact classical solver |
| `solve_min_conflicts()`  | Min-conflicts heuristic solver |
| `grover_demo_example()`  | 3-qubit Grover demo with histogram |
| `plot_schedule()`        | Visual grid-based exam schedule plot |
| `main.py` / runner       | Runs solvers + demo for N=8 by default |

---

## Running the Project  
### **Install Dependencies**
```bash
pip install matplotlib qiskit qiskit-aer
Run the Example
bash
Copy code
python main.py
This runs:

Backtracking

Min-conflicts

Grover’s quantum demo (if Qiskit installed)

Purpose (Course Project Requirement)
This project is submitted as part of the BITE4077L Quantum Computing course.
It demonstrates:

Mapping exam scheduling to constraint-satisfaction

Comparing classical (backtracking, heuristics) and quantum search

Implementing Grover’s algorithm on simulated quantum hardware

Creating a hybrid classical-quantum workflow
