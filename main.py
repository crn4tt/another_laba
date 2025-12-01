from tkinter import *
from tkinter import ttk
import matplotlib as plt
plt.use("tkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import algs
import scipy
import scipy.optimize

def get_values():
    fig.clear()
    plot1 = fig.add_subplot()
    values = dict()
    values["batch_count"] = int(bc_entry.get())
    values["min_sug"] = float(st_entry_min.get())
    values["max_sug"] = float(st_entry_max.get())
    values["min_deg"] = float(degr_entry_min.get())
    values["max_deg"] = float(degr_entry_max.get())
    values["dop_cond"] = enabled.get()
    values["rip_per"] = int(dozar_entry.get())
    values["minK"] = 4.8
    values["maxK"] = 7.05
    values["minNa"] = 0.21
    values["maxNa"] = 0.28
    values["minN"] = 1.58
    values["maxN"] = 2.8
    values["minRip"] = float(dc_entry_min.get())
    values["maxRip"] = float(dc_entry_max.get())
    run(values, plot1)


def run(values: dict, plot1):
    count_of_exp = int(ne_entry.get())
    coef_rip = algs.gen_random_matrix(values["batch_count"], values["rip_per"], values["minRip"], values["maxRip"])
    coef_deg = algs.gen_random_matrix(values["batch_count"], values["batch_count"] - values["rip_per"],
                                      values["minRip"], values["maxRip"])
    matr = algs.merge_matrices(coef_rip, coef_deg)
    matr_to_algs = algs.gen_random_base_matrix(values["batch_count"], values["min_sug"], values["max_sug"], matr)
    if values["dop_cond"]:
        inorganic_matr = algs.gen_inorganic_matrix(values["batch_count"], values["minK"], values["maxK"],
                                                   values["minNa"], values["maxNa"], values["minN"], values["maxN"])
        matr_to_algs = algs.get_inorganic(matr_to_algs, inorganic_matr)

    days = [i for i in range(1, values["batch_count"] + 1)]
    hun_mean = [0 for i in range(values["batch_count"])]
    gready_mean = [0 for i in range(values["batch_count"])]
    thrifty_mean = [0 for i in range(values["batch_count"])]
    gvt_mean = [0 for i in range(values["batch_count"])]
    tvg_mean = [0 for i in range(values["batch_count"])]
    results = dict()

    for i in range(count_of_exp):

        m = algs.hungarian_max_algorithm(matr_to_algs)[0]
        for j in range(len(m)): hun_mean[j] += m[j] / count_of_exp
        results["венгерский максимальный"] = hun_mean[len(m) - 1]

        m = algs.greedy_algorithm(matr_to_algs)[0]
        for j in range(len(m)): gready_mean[j] += m[j] / count_of_exp
        results["жадный"] = gready_mean[len(m) - 1]

        m = algs.thrifty_algorithm(matr_to_algs)[0]
        for j in range(len(m)): thrifty_mean[j] += m[j] / count_of_exp
        results["бережливый"] = thrifty_mean[len(m) - 1]

        m = algs.greedy_v_thrifty_algorithm(matr_to_algs, values["rip_per"])[0]
        for j in range(len(m)): gvt_mean[j] += m[j] / count_of_exp
        results["жадный/бережливый"] = gvt_mean[len(m) - 1]

        m = algs.thrifty_v_greedy_algorithm(matr_to_algs, values["rip_per"])[0]
        for j in range(len(m)): tvg_mean[j] += m[j] / count_of_exp
        results["бережливый/жадный"] = tvg_mean[len(m) - 1]

    plot1.clear()
    plot1.plot(days, hun_mean, label="Венгреский максимальный", color="#f74040")
    plot1.plot(days, gready_mean, label="Жадный", color="#a334f7")
    plot1.plot(days, thrifty_mean, label="Бережливый", color="#14b54a")
    plot1.plot(days, gvt_mean, label="Жадный/Бережливый", color="#e8a51e")
    plot1.plot(days, tvg_mean, label="Бережлиый/Жадный", color="#fa7528")
    plot1.legend()
    canvas.draw()



    for key in results.keys():
        if results[key] == max(results.values()):
            good["text"] = f"Лучший результат выдал {key} алгоритм"
        if results[key] == min(results.values()):
            bad["text"] = f"Худший результат выдал {key} алгоритм"

window = Tk()
window.title("Рассчёт сахаризации свеклы")
windowWidth = int(window.winfo_screenwidth() * 0.8)
windowHeight = int(window.winfo_screenheight() * 0.8)
window.geometry(f"{windowWidth}x{windowHeight}")

style = ttk.Style()
style.theme_use("clam")

#notebook
notebookWidth = int(windowWidth * 0.95)
notebookHeight = int(windowHeight * 0.9)
notebook = ttk.Notebook(width=notebookWidth, height=notebookHeight)
notebook.place(relx=0.5, rely=0.5, anchor="center")
style.configure("TNotebook.Tab", font=("Arial", 15))
style.configure("TButton", font=("Arial, 30"), justify="center")

bg1 = Frame(notebook)
bg1.pack(fill=BOTH)
bg1.pack_propagate(False)
notebook.add(bg1, text="Конфигурация")

#batch count frame
bc_frame_w = int(windowWidth*0.4)
bc_frame_h = int(windowHeight*0.15)
bc_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=bc_frame_w, height=bc_frame_h, padding=[8, 10])

bc_label = ttk.Label(bc_frame, text="Количество партий", font="Arial 30")
bc_entry = ttk.Entry(bc_frame, width=int(bc_frame_w * 0.01), font="Arial 30")
bc_entry.configure(justify="center")

#dozar count frame
dozar_frame_w = int(windowWidth*0.4)
dozar_frame_h = int(windowHeight*0.15)
dozar_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=dozar_frame_w, height=dozar_frame_h, padding=[8, 10])

dozar_label = ttk.Label(dozar_frame, text="Период дозаривания", font="Arial 30")
dozar_entry = ttk.Entry(dozar_frame, width=int(dozar_frame_w * 0.01), font="Arial 30")
dozar_entry.configure(justify="center")

#degr diap
degr_frame_w = int(windowWidth*0.4)
degr_frame_h = int(windowHeight*0.15)
degr_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=degr_frame_w, height=degr_frame_h, padding=[8, 10])

degr_label = ttk.Label(degr_frame, text="Диапазон деградации", font="Arial 30")
degr_entry_min = ttk.Entry(degr_frame, width=int(degr_frame_w * 0.01), font="Arial 30")
degr_entry_max = ttk.Entry(degr_frame, width=int(degr_frame_w * 0.01), font="Arial 30")
degr_entry_min.configure(justify="center")
degr_entry_max.configure(justify="center")
for i in range(2): degr_frame.columnconfigure(index=i, weight=1)
for i in range(2): degr_frame.rowconfigure(index=i, weight=1)

#coef_dozar
dc_frame_w = int(windowWidth*0.4)
dc_frame_h = int(windowHeight*0.15)
dc_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=dc_frame_w, height=dc_frame_h, padding=[8, 10])

dc_label = ttk.Label(dc_frame, text="Диапазон коэф. дозаривания", font="Arial 30")
dc_entry_min = ttk.Entry(dc_frame, width=int(dc_frame_w * 0.01), font="Arial 30")
dc_entry_max = ttk.Entry(dc_frame, width=int(dc_frame_w * 0.01), font="Arial 30")
dc_entry_min.configure(justify="center")
dc_entry_max.configure(justify="center")
for i in range(2): dc_frame.columnconfigure(index=i, weight=1)
for i in range(2): dc_frame.rowconfigure(index=i, weight=1)

#start suger
st_frame_w = int(windowWidth*0.4)
st_frame_h = int(windowHeight*0.15)
st_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=st_frame_w, height=st_frame_h, padding=[8, 10])

st_label = ttk.Label(st_frame, text="Диапазон нач. сахаристости", font="Arial 30")
st_entry_min = ttk.Entry(st_frame, width=int(st_frame_w * 0.01), font="Arial 30")
st_entry_max = ttk.Entry(st_frame, width=int(st_frame_w * 0.01), font="Arial 30")
st_entry_min.configure(justify="center")
st_entry_max.configure(justify="center")
for i in range(2): st_frame.columnconfigure(index=i, weight=1)
for i in range(2): st_frame.rowconfigure(index=i, weight=1)

#num exp
ne_frame_w = int(windowWidth*0.4)
ne_frame_h = int(windowHeight*0.15)
ne_frame = ttk.Frame(bg1, borderwidth=1, relief=RIDGE,
                     width=ne_frame_w, height=ne_frame_h, padding=[8, 10])

ne_label = ttk.Label(ne_frame, text="Количество эксперементов", font="Arial 30")
ne_entry = ttk.Entry(ne_frame, width=int(ne_frame_w * 0.01), font="Arial 30")
ne_entry.configure(justify="center")

#dop cond
enabled = IntVar()
dop_cond = Checkbutton(bg1, text="Учитывать доп. условия", variable=enabled, font="Arial 30")

for i in range(2): bg1.columnconfigure(index=i, weight=1)
for i in range(4): bg1.rowconfigure(index=i, weight=1)

#button
button = ttk.Button(bg1, text="Рассчитать", width=int(windowWidth * 0.01), command=get_values)

#graph
bg2 = Frame(notebook)
bg2.pack(fill=BOTH)
bg2.pack_propagate(False)
notebook.add(bg2, text="График")
fig = Figure(figsize=(15, 10), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=bg2)
canvas.get_tk_widget().pack()
plot1 = fig.add_subplot()

#itog
bg3 = Frame(notebook)
bg3.pack(fill=BOTH)
bg3.pack_propagate(False)
notebook.add(bg3, text="Интерпретация")
lw = int(windowWidth*0.4)
lh = int(windowHeight*0.15)
good_frame = ttk.Frame(bg3, borderwidth=1, relief=RIDGE,
                       width=lw, height=lh, padding=[8, 10])
bad_frame = ttk.Frame(bg3, borderwidth=1, relief=RIDGE,
                       width=lw, height=lh, padding=[8, 10])
good = ttk.Label(good_frame, font="Arial 30")
bad = ttk.Label(bad_frame, font="Arial 30")
for i in range(1): bg3.columnconfigure(index=i, weight=1)
for i in range(2): bg3.rowconfigure(index=i, weight=1)


bc_frame.grid(row=0, column=0)
bc_frame.pack_propagate(False)
dozar_frame.grid(row=0, column=1)
dozar_frame.pack_propagate(False)
degr_frame.grid(row=1, column=0)
degr_frame.grid_propagate(False)
dc_frame.grid(row=1, column=1)
dc_frame.grid_propagate(False)
st_frame.grid(row=2, column=0)
st_frame.grid_propagate(False)
ne_frame.grid(row=2, column=1)
ne_frame.pack_propagate(False)
dop_cond.grid(row=3, column=0)
button.grid(row=3, column=1)
bc_label.pack()
bc_entry.pack()
ne_label.pack()
ne_entry.pack()
dozar_label.pack()
dozar_entry.pack()
degr_label.grid(row=0, columnspan=2)
degr_entry_min.grid(row=1, column=0)
degr_entry_max.grid(row=1, column=1)
dc_label.grid(row=0, columnspan=2)
dc_entry_min.grid(row=1, column=0)
dc_entry_max.grid(row=1, column=1)
st_label.grid(row=0, columnspan=2)
st_entry_min.grid(row=1, column=0)
st_entry_max.grid(row=1, column=1)
good_frame.grid(row=0, column=0)
good_frame.grid_propagate(False)
bad_frame.grid(row=1, column=0)
bad_frame.grid_propagate(False)
good.pack()
bad.pack()

window.mainloop()
