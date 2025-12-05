from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List

import flet as ft
from flet.plotly_chart import PlotlyChart
import plotly.graph_objects as go

from . import (
    adjust_for_inorganic,
    base_sugar_matrix,
    greedy_algorithm,
    greedy_then_thrifty,
    hungarian_max_algorithm,
    inorganic_matrix,
    calculate_losses_matrix,
    merge_matrices,
    random_matrix,
    concentrated_matrix,
    thrifty_algorithm,
    thrifty_then_greedy,
)

MatrixSummary = Dict[str, List[float]]


@dataclass
class SimulationConfig:
    batches: int
    ripening_period: int
    min_sugar: float
    max_sugar: float

    min_rip_coeff: float
    max_rip_coeff: float
    min_deg_coeff: float
    max_deg_coeff: float

    include_inorganic: bool
    include_ripening: bool
    experiments: int
    dist_type: str
    daily_tonnage: float

    min_k: float
    max_k: float
    min_na: float
    max_na: float
    min_n: float
    max_n: float
    min_i0: float
    max_i0: float


class SugarBeetApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.page.title = "Рассчёт сахаризации свеклы"
        self.page.theme = ft.Theme(color_scheme_seed=ft.Colors.TEAL_400)
        self.page.bgcolor = "#F5F7F8"
        self.page.padding = 0
        self.page.scroll = None

        self.fields: Dict[str, ft.TextField] = {}
        self.results_cache: Dict[str, float] = {}

        self.include_inorganic = ft.Switch(
            value=False,
            active_color=ft.Colors.TEAL_600,
            on_change=self._toggle_inorganic_fields
        )

        self.include_ripening = ft.Switch(
            value=True,
            active_color=ft.Colors.TEAL_600,
            on_change=self._toggle_ripening_fields
        )

        self.dist_group = ft.RadioGroup(
            content=ft.Column([
                ft.Radio(
                    value="uniform",
                    label="Равномерное",
                    label_style=ft.TextStyle(color=ft.Colors.GREY_800, size=14)),
                ft.Radio(
                    value="concentrated",
                    label="Концентрированное",
                    label_style=ft.TextStyle(color=ft.Colors.GREY_800, size=14)),
            ]),
            value="uniform"
        )

        self.chart = PlotlyChart(self._build_chart_figure(show_annotation=True), expand=True)
        self.summary_table = self._build_summary_table({})

        self.recommendation_text = ft.Text(value="", size=16, color=ft.Colors.GREY_800)
        self.recommendation_container = ft.Container(
            content=ft.Column([
                ft.Text(
                    "Рекомендация СППР",
                    weight=ft.FontWeight.BOLD,
                    size=18,
                    color=ft.Colors.TEAL_800),
                ft.Divider(height=10, color=ft.Colors.TEAL_200),
                self.recommendation_text
            ]),
            padding=20,
            bgcolor=ft.Colors.TEAL_50,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.TEAL_200),
            visible=False
        )

        self.best_text = ft.Text(
            weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_700, size=16)
        self.worst_text = ft.Text(
            weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700, size=16)

        self.progress_bar = ft.ProgressBar(
            width=200, color=ft.Colors.TEAL, bgcolor=ft.Colors.TEAL_100)
        self.loading_text = ft.Text("Выполняется расчёт...", size=16, color=ft.Colors.GREY_800)
        self.loading_overlay = ft.Container(visible=False)
        self.btn_run = ft.ElevatedButton()

        self.tabs: ft.Tabs | None = None
        self.inorganic_params_container = ft.Column(visible=False, spacing=12)
        self.ripening_params_container = ft.Column(visible=True, spacing=12)

        self._build_layout()

    def _loss_chart_placeholder(self):
        return PlotlyChart(self._build_loss_figure({}), expand=True)

    def _build_layout(self) -> None:
        sidebar = ft.Container(
            content=self._build_sidebar_content(),
            width=380,
            padding=25,
            bgcolor=ft.Colors.WHITE,
            shadow=ft.BoxShadow(
                blur_radius=10,
                color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
                offset=ft.Offset(2, 0)),
        )

        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ANALYTICS, size=32, color=ft.Colors.TEAL_800),
                    ft.Text(
                        "Аналитика сахаристости",
                        size=28,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREY_800),
                ],
            ),
            padding=ft.padding.only(left=20, top=20, bottom=10),
        )

        self.loss_chart = PlotlyChart(self._build_loss_figure({}), expand=True)

        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            indicator_color=ft.Colors.TEAL,
            label_color=ft.Colors.TEAL,
            unselected_label_color=ft.Colors.GREY_600,
            tabs=[
                ft.Tab(
                    text="График",
                    icon=ft.Icons.SHOW_CHART,
                    content=ft.Container(content=self.chart, padding=10),
                ),
                ft.Tab(
                    text="Таблица",
                    icon=ft.Icons.TABLE_CHART,
                    content=ft.Column(
                        controls=[
                            self.summary_table,
                            ft.Container(height=20),
                            self.recommendation_container
                        ],
                        scroll=ft.ScrollMode.AUTO,
                        expand=True
                    ),
                ),
                ft.Tab(
                    text="Потери",
                    icon=ft.Icons.BAR_CHART,
                    content=ft.Container(content=self.loss_chart, padding=10),
                ),
            ],
            expand=True,
        )

        metrics_row = ft.Container(
            padding=ft.padding.symmetric(horizontal=20),
            content=ft.Row(
                controls=[
                    self._build_metric_card(
                        "Лучший алгоритм",
                        self.best_text,
                        ft.Icons.TRENDING_UP,
                        ft.Colors.GREEN_50),
                    self._build_metric_card(
                        "Худший алгоритм",
                        self.worst_text,
                        ft.Icons.TRENDING_DOWN,
                        ft.Colors.RED_50),
                ],
                spacing=20,
            )
        )

        results_content = ft.Column(
            controls=[
                metrics_row,
                ft.Container(
                    content=self.tabs,
                    expand=True,
                    padding=ft.padding.only(left=20, right=20, bottom=20))
            ],
            expand=True,
        )

        self.loading_overlay = ft.Container(
            content=ft.Column(
                controls=[self.progress_bar, ft.Container(height=20), self.loading_text],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.with_opacity(0.9, "#F5F7F8"),
            alignment=ft.alignment.center,
            visible=False,
            expand=True,
        )

        main_stack = ft.Stack([results_content, self.loading_overlay], expand=True)
        main_area = ft.Column([header, main_stack], expand=True, spacing=0)

        self.page.add(ft.Row([sidebar, main_area], expand=True, spacing=0))

    def _build_sidebar_content(self) -> ft.Column:
        self.btn_run = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.ROCKET_LAUNCH, size=22, color=ft.Colors.WHITE),
                 ft.Text(
                     "Запустить расчёт",
                     size=18,
                     weight=ft.FontWeight.W_600,
                     color=ft.Colors.WHITE)],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            on_click=self._handle_run,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=22,
                bgcolor={
                    "": ft.Colors.TEAL_600,
                    "hovered": ft.Colors.TEAL_500,
                    "disabled": ft.Colors.GREY_400
                },
                color={"": ft.Colors.WHITE, "disabled": ft.Colors.WHITE54},
                elevation={"press": 2, "": 6},
                animation_duration=300,
            ),
            width=float("inf")
        )

        btn_reset = ft.TextButton(
            content=ft.Row([
                ft.Icon(ft.Icons.REFRESH, size=18, color=ft.Colors.TEAL),
                ft.Text("Сбросить параметры", size=16, color=ft.Colors.TEAL)
            ], alignment=ft.MainAxisAlignment.CENTER),
            on_click=self._reset_fields,
            width=float("inf")
        )

        self.inorganic_params_container.controls = [
            ft.Text(
                "K (ммоль/100г)", size=14, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_800),
            ft.Row([
                    self._number_field("Мин", "4.8", "min_k"),
                    self._number_field("Макс", "7.05", "max_k")],
                spacing=10
            ),

            ft.Text(
                "Na (ммоль/100г)", size=14, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_800),
            ft.Row([
                    self._number_field("Мин", "0.21", "min_na"),
                    self._number_field("Макс", "0.82", "max_na")],
                spacing=10),

            ft.Text(
                "N (ммоль/100г)", size=14, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_800),
            ft.Row([
                    self._number_field("Мин", "1.58", "min_n"),
                    self._number_field("Макс", "2.8", "max_n")],
                spacing=10),

            ft.Text(
                "I0 (Редуцирующие, %)",
                size=14,
                weight=ft.FontWeight.W_500,
                color=ft.Colors.GREY_800),
            ft.Row([
                    self._number_field("Мин", "0.62", "min_i0"),
                    self._number_field("Макс", "0.64", "max_i0")],
                spacing=10),
        ]

        self.ripening_params_container.controls = [
             ft.Container(height=5),
             self._number_field("Длительность дозаривания (v)", "7", "ripening"),

             ft.Text(
                 "Коэффициенты роста (> 1)",
                 size=14,
                 color=ft.Colors.TEAL_700,
                 weight=ft.FontWeight.W_500),
             ft.Row([
                self._number_field("Мин", "1.01", "min_rip"),
                self._number_field("Макс", "1.15", "max_rip"),
             ], spacing=10),
        ]

        general_section = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(
                        ft.Icons.SETTINGS, color=ft.Colors.TEAL_700, size=24),
                    ft.Text(
                        "Общие параметры",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREY_800),
                ], spacing=10),

                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),

                self._number_field("Кол-во партий (n)", "15", "batches"),
                self._number_field("Сут. переработка (т)", "3000", "tonnage"),
                self._number_field("Эксперименты", "50", "experiments"),
            ], spacing=10),
            bgcolor=ft.Colors.GREY_50,
            padding=15,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_300)
        )

        raw_material_section = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ECO, color=ft.Colors.TEAL_700, size=24),
                    ft.Text("Сырье", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ], spacing=10),

                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),

                ft.Text(
                    "Сахаристость (%)",
                    size=15,
                    color=ft.Colors.GREY_800,
                    weight=ft.FontWeight.W_500),
                ft.Row([
                    self._number_field("Мин.", "12", "min_sugar"),
                    self._number_field("Макс.", "22", "max_sugar"),
                ], spacing=10),
            ], spacing=10),
            bgcolor=ft.Colors.GREY_50,
            padding=15,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_300)
        )

        dynamics_section = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(
                        ft.Icons.TIMELINE, color=ft.Colors.TEAL_700, size=24),
                    ft.Text(
                        "Динамика сахаристости",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREY_800),
                ], spacing=10),

                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),

                ft.Row([
                    ft.Text(
                        "Учитывать дозаривание",
                        size=15,
                        weight=ft.FontWeight.W_500,
                        color=ft.Colors.GREY_800,
                        expand=True),
                    self.include_ripening
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                self.ripening_params_container,

                ft.Divider(height=20, thickness=1, color=ft.Colors.GREY_300),

                ft.Text(
                    "Коэффициенты деградации (< 1)",
                    size=14,
                    weight=ft.FontWeight.W_500,
                    color=ft.Colors.RED_700),
                ft.Row([
                    self._number_field("Мин", "0.85", "min_deg"),
                    self._number_field("Макс", "0.99", "max_deg"),
                ], spacing=10),

                ft.Divider(height=20, thickness=1, color=ft.Colors.GREY_300),

                ft.Text(
                    "Тип распределения коэффициентов",
                    size=15,
                    color=ft.Colors.GREY_800,
                    weight=ft.FontWeight.W_500),
                self.dist_group

            ], spacing=10),
            bgcolor=ft.Colors.GREY_50,
            padding=15,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_300)
        )

        inorganic_section = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCIENCE, color=ft.Colors.TEAL_700, size=24),
                    ft.Text(
                        "Неорганические соединения",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREY_800),
                ], spacing=10),

                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),

                ft.Row([
                    ft.Text(
                        "Учитывать при расчете",
                        size=15,
                        weight=ft.FontWeight.W_500,
                        color=ft.Colors.GREY_800,
                        expand=True),
                    self.include_inorganic
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.inorganic_params_container
            ], spacing=10),
            bgcolor=ft.Colors.GREY_50,
            padding=15,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_300)
        )

        return ft.Column(
            controls=[
                ft.Text("Параметры", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Divider(color=ft.Colors.TRANSPARENT, height=10),

                general_section,
                ft.Container(height=10),

                raw_material_section,
                ft.Container(height=10),

                dynamics_section,
                ft.Container(height=10),

                inorganic_section,

                ft.Container(expand=True),

                self.btn_run,
                ft.Container(height=5),
                btn_reset
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8
        )

    def _number_field(self, label: str, value: str, key: str) -> ft.Control:
        tf = ft.TextField(
            label=label,
            value=value,
            keyboard_type=ft.KeyboardType.NUMBER,
            border_radius=8,
            dense=True,
            filled=True,
            bgcolor=ft.Colors.GREY_50,
            border_color=ft.Colors.TRANSPARENT,
            text_style=ft.TextStyle(color=ft.Colors.GREY_800, size=16),
            label_style=ft.TextStyle(color=ft.Colors.GREY_700, size=16),
            focused_border_color=ft.Colors.TEAL,
            content_padding=12,
            height=45
        )
        self.fields[key] = tf
        return ft.Container(content=tf, expand=1)

    def _build_metric_card(self, title, control, icon, bg_color):
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Icon(icon, color=ft.Colors.BLACK54, size=24),
                    padding=12,
                    bgcolor=bg_color,
                    border_radius=10,
                ),
                ft.Column([
                    ft.Text(title, size=14, color=ft.Colors.GREY_700),
                    control
                ], spacing=2)
            ]),
            bgcolor=ft.Colors.WHITE,
            padding=15,
            border_radius=12,
            shadow=ft.BoxShadow(
                blur_radius=5,
                color=ft.Colors.with_opacity(0.05, ft.Colors.BLACK),
                offset=ft.Offset(0, 2)
            ),
            expand=True
        )

    def _build_loss_figure(self, final_values: Dict[str, float]) -> go.Figure:
        fig = go.Figure()

        if not final_values:
            fig.update_layout(
                template="plotly_white",
                annotations=[dict(
                    text="Запустите расчёт",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=18, color="gray")
                )]
            )
            return fig

        hungarian = final_values["Венгерский (макс.)"]

        algorithms = []
        losses = []

        for name, val in final_values.items():
            if "Венгерский" in name:
                continue
            algorithms.append(name)
            losses.append((hungarian - val) / hungarian * 100)

        fig.add_trace(go.Bar(
            x=algorithms,
            y=losses,
            text=[f"{l:.2f}%" for l in losses],
            textposition="outside",
            marker=dict(color="#26a69a")
        ))

        fig.update_layout(
            template="plotly_white",
            yaxis_title="Потери (%)",
            xaxis_title="Алгоритм",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        return fig


    def _toggle_inorganic_fields(self, e):
        self.inorganic_params_container.visible = self.include_inorganic.value
        self.inorganic_params_container.update()

    def _toggle_ripening_fields(self, e):
        self.ripening_params_container.visible = self.include_ripening.value
        self.ripening_params_container.update()

    def _handle_run(self, _: ft.ControlEvent) -> None:
        try:
            config = self._parse_config()
        except ValueError as exc:
            self._toast(str(exc))
            return

        self._toggle_loading(True)
        time.sleep(0.1)
        threading.Thread(target=self._run_simulation_thread, args=(config,), daemon=True).start()

    def _run_simulation_thread(self, config: SimulationConfig):
        try:
            raw_averages = self._run_simulation(config)

            tonnage_averages = {}
            for name, values in raw_averages.items():
                tonnage_averages[name] = [(val / 100.0) * config.daily_tonnage for val in values]

            self.results_cache = {name: vals[-1] for name, vals in tonnage_averages.items()}

            self._update_chart(list(range(1, config.batches + 1)), tonnage_averages)
            self._update_summary(tonnage_averages)
            self._update_recommendation(tonnage_averages)

            self._toggle_loading(False)

        except Exception as exc:
            print(f"Error in simulation thread: {exc}")
            self._toggle_loading(False)

    def _toggle_loading(self, is_loading: bool):
        self.loading_overlay.visible = is_loading
        self.btn_run.disabled = is_loading
        self.loading_overlay.update()
        self.btn_run.update()

    def _parse_config(self) -> SimulationConfig:
        def get_val(key, type_func=float):
            val = self.fields[key].value
            if not val: raise ValueError(f"Поле пустое")
            return type_func(val)

        try:
            return SimulationConfig(
                batches=get_val("batches", int),
                ripening_period=get_val("ripening", int),
                min_sugar=get_val("min_sugar"),
                max_sugar=get_val("max_sugar"),

                min_rip_coeff=get_val("min_rip"),
                max_rip_coeff=get_val("max_rip"),
                min_deg_coeff=get_val("min_deg"),
                max_deg_coeff=get_val("max_deg"),

                include_inorganic=self.include_inorganic.value,
                include_ripening=self.include_ripening.value,
                experiments=get_val("experiments", int),
                dist_type=self.dist_group.value,
                daily_tonnage=get_val("tonnage"),

                min_k=get_val("min_k"), max_k=get_val("max_k"),
                min_na=get_val("min_na"), max_na=get_val("max_na"),
                min_n=get_val("min_n"), max_n=get_val("max_n"),
                min_i0=get_val("min_i0"), max_i0=get_val("max_i0"),
            )
        except ValueError as e:
            raise ValueError("Проверьте корректность введенных чисел.") from e

    def _run_simulation(self, config: SimulationConfig) -> MatrixSummary:
        gen_func = random_matrix if config.dist_type == "uniform" else concentrated_matrix

        averages = {
            "Венгерский (макс.)": [0.0] * config.batches,
            "Жадный": [0.0] * config.batches,
            "Бережливый": [0.0] * config.batches,
            "Жадный -> Бережливый": [0.0] * config.batches,
            "Бережливый -> Жадный": [0.0] * config.batches,
        }

        for _ in range(config.experiments):
            if config.include_ripening:
                coef_ripening = gen_func(
                    config.batches,
                    config.ripening_period,
                    config.min_rip_coeff,
                    config.max_rip_coeff,
                )
                coef_degradation = gen_func(
                    config.batches,
                    config.batches - config.ripening_period,
                    config.min_deg_coeff,
                    config.max_deg_coeff,
                )
                coefficients = merge_matrices(coef_ripening, coef_degradation)
            else:
                coefficients = gen_func(
                    config.batches,
                    config.batches,
                    config.min_deg_coeff,
                    config.max_deg_coeff,
                )

            base_matrix = base_sugar_matrix(
                config.batches,
                config.min_sugar,
                config.max_sugar,
                coefficients,
            )

            if config.include_inorganic:
                inorganic = inorganic_matrix(
                    config.batches,
                    config.min_k, config.max_k,
                    config.min_na, config.max_na,
                    config.min_n, config.max_n,
                )
                losses = calculate_losses_matrix(
                    config.batches,
                    inorganic,
                    config.min_i0,
                    config.max_i0,
                )
                working_matrix = adjust_for_inorganic(base_matrix, losses)
            else:
                working_matrix = base_matrix

            self._accumulate_results(
                averages["Венгерский (макс.)"],
                hungarian_max_algorithm(working_matrix)[0],
                config.experiments,
            )
            self._accumulate_results(
                averages["Жадный"],
                greedy_algorithm(working_matrix)[0],
                config.experiments,
            )
            self._accumulate_results(
                averages["Бережливый"],
                thrifty_algorithm(working_matrix)[0],
                config.experiments,
            )
            self._accumulate_results(
                averages["Жадный -> Бережливый"],
                greedy_then_thrifty(working_matrix, config.ripening_period)[0],
                config.experiments,
            )
            self._accumulate_results(
                averages["Бережливый -> Жадный"],
                thrifty_then_greedy(working_matrix, config.ripening_period)[0],
                config.experiments,
            )

        return averages

    def _accumulate_results(
        self, target: List[float], values: List[float], experiments: int
    ) -> None:
        for idx, value in enumerate(values):
            target[idx] += value / experiments

    def _build_chart_figure(self, show_annotation: bool = False) -> go.Figure:
        fig = go.Figure()
        annotations_list = []
        if show_annotation:
            annotations_list.append(dict(
                text="Нажмите 'Запустить расчёт'", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font=dict(color="gray", size=18),
            ))

        fig.update_layout(
            template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="День", yaxis_title="Кумулятивный показатель",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=annotations_list,
            font=dict(size=14)
        )
        return fig

    def _update_chart(self, days: List[int], averages: MatrixSummary) -> None:
        fig = self._build_chart_figure(show_annotation=False)
        palette = ["#26a69a", "#ec407a", "#66bb6a", "#ffa726", "#ab47bc"]

        for i, (name, values) in enumerate(averages.items()):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=days, y=values, mode="lines+markers", name=name,
                line=dict(color=color, width=3, shape='spline'),
                marker=dict(size=8, line=dict(width=2, color='white')),
            ))
        self.chart.figure = fig
        self.chart.update()

    def _build_summary_table(self, final_values: Dict[str, float]) -> ft.DataTable:
        rows = []
        if final_values:
            max_val = max(final_values.values())
            sorted_items = sorted(final_values.items(), key=lambda item: item[1], reverse=True)
            for i, (name, score) in enumerate(sorted_items):
                is_best = i == 0
                loss_pct = ((max_val - score) / max_val) * 100 if max_val != 0 else 0

                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Row([
                        ft.Icon(
                            ft.Icons.EMOJI_EVENTS if is_best else ft.Icons.CIRCLE,
                            size=18,
                            color=ft.Colors.AMBER if is_best else ft.Colors.GREY_300),
                        ft.Text(
                            name,
                            weight=ft.FontWeight.BOLD if is_best else ft.FontWeight.NORMAL,
                            color=ft.Colors.GREY_800,
                            size=14)
                    ], spacing=10)),
                    ft.DataCell(ft.Text(
                        f"{score:,.0f} т",
                        color=ft.Colors.GREEN_700 if is_best else ft.Colors.GREY_800,
                        size=14)),
                    ft.DataCell(ft.Container(
                        content=ft.Text(
                            f"-{loss_pct:.2f}%",
                            color=ft.Colors.WHITE,
                            size=13,
                            weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.GREEN_200 if loss_pct < 0.01 else (
                            ft.Colors.ORANGE_200 if loss_pct < 5 else ft.Colors.RED_200),
                        padding=ft.padding.symmetric(horizontal=10, vertical=4),
                        border_radius=6,
                        alignment=ft.alignment.center
                    )),
                ]))

        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text(
                    "Алгоритм", color=ft.Colors.GREY_800, size=15, weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text(
                        "Сахар (тонн)",
                        text_align=ft.TextAlign.RIGHT,
                        color=ft.Colors.GREY_800,
                        size=15,
                        weight=ft.FontWeight.BOLD),
                    numeric=True),
                ft.DataColumn(ft.Text(
                        "Потери",
                        text_align=ft.TextAlign.RIGHT,
                        color=ft.Colors.GREY_800,
                        size=15,
                        weight=ft.FontWeight.BOLD),
                    numeric=True),
            ],
            rows=rows,
            border_radius=10,
            column_spacing=20,
            width=float("inf"),
            heading_row_color=ft.Colors.GREY_200,
            data_row_color=ft.Colors.WHITE,
            divider_thickness=0.5
        )

    def _update_summary(self, averages: MatrixSummary) -> None:
        if not averages:
            return
        final_values = {name: values[-1] for name, values in averages.items()}
        best_name = max(final_values, key=final_values.get)
        worst_name = min(final_values, key=final_values.get)

        self.best_text.value = f"{best_name} ({final_values[best_name]:,.0f} т)"
        self.worst_text.value = f"{worst_name} ({final_values[worst_name]:,.0f} т)"
        self.best_text.update()
        self.worst_text.update()

        self.summary_table.rows = self._build_summary_table(final_values).rows
        self.summary_table.update()

        self.loss_chart.figure = self._build_loss_figure(final_values)
        self.loss_chart.update()

    def _update_recommendation(self, averages: MatrixSummary) -> None:
        if not averages:
            return

        final_values = {name: values[-1] for name, values in averages.items()}
        real_strategies = {k: v for k, v in final_values.items() if "Венгерский" not in k}

        if not real_strategies:
            return

        best_strat = max(real_strategies, key=real_strategies.get)
        ideal_val = final_values.get("Венгерский (макс.)", list(real_strategies.values())[0])
        best_val = real_strategies[best_strat]

        loss = 0.0
        if ideal_val != 0:
            loss = ((ideal_val - best_val) / ideal_val) * 100

        rec_msg = (f"Рекомендуемая стратегия: {best_strat}\n"
                   f"Потери относительно эталона: {loss:.2f}%")

        self.recommendation_text.value = rec_msg
        self.recommendation_container.visible = True
        self.recommendation_container.update()

    def _reset_fields(self, _: ft.ControlEvent) -> None:
        defaults = {
            "batches": "15", "ripening": "7", "min_sugar": "12", "max_sugar": "22",
            "min_rip": "1.01", "max_rip": "1.15", "min_deg": "0.85", "max_deg": "0.99",
            "experiments": "50", "tonnage": "3000",
            "min_k": "4.8", "max_k": "7.05", "min_na": "0.21", "max_na": "0.82",
            "min_n": "1.58", "max_n": "2.8", "min_i0": "0.62", "max_i0": "0.64"
        }
        for key, value in defaults.items():
            if key in self.fields:
                self.fields[key].value = value

        self.include_inorganic.value = False
        self.include_ripening.value = True
        self.dist_group.value = "uniform"
        self.inorganic_params_container.visible = False
        self.ripening_params_container.visible = True

        self.best_text.value = ""
        self.worst_text.value = ""
        self.chart.figure = self._build_chart_figure(show_annotation=True)
        self.summary_table.rows = []
        self.recommendation_container.visible = False

        self.summary_table.update()
        self.chart.update()
        self.best_text.update()
        self.worst_text.update()
        self.recommendation_container.update()
        if self.tabs:
            self.tabs.update()

        for field in self.fields.values():
            field.update()
        self.include_inorganic.update()
        self.include_ripening.update()
        self.inorganic_params_container.update()
        self.ripening_params_container.update()
        self.dist_group.update()

    def _toast(self, message: str) -> None:
        self.page.snack_bar = ft.SnackBar(
            ft.Text(message), bgcolor=ft.Colors.RED_300, duration=3000)
        self.page.snack_bar.open = True
        self.page.update()

def launch(page: ft.Page) -> None:
    SugarBeetApp(page)

__all__ = ["SugarBeetApp", "launch"]
