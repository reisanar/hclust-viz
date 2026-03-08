from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage

from manim import (
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    WHITE,
    YELLOW,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    Axes,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)


@dataclass
class PointRow:
    idx: int
    x: float
    y: float
    species: str


def load_points(path: str) -> List[PointRow]:
    pts: List[PointRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append(
                PointRow(
                    idx=int(row["id"]),
                    x=float(row["petal_length_cm"]),
                    y=float(row["petal_width_cm"]),
                    species=row["species"],
                )
            )
    return pts


class HierarchicalClusteringDemo(Scene):
    """A step-by-step animation of agglomerative hierarchical clustering and dendrogram construction."""

    def construct(self):
        # ----------------------
        # Data and clustering
        # ----------------------
        pts = load_points("data/iris_8.csv")
        X = np.array([[p.x, p.y] for p in pts], dtype=float)

        # Average linkage is easy to explain: distance between clusters = average of pairwise distances
        Z = linkage(X, method="average", metric="euclidean")
        # Z has shape (n-1, 4): [cluster_a, cluster_b, distance, new_cluster_size]

        # ----------------------
        # Layout: scatter (left) + dendrogram (right)
        # ----------------------
        title = Text("Hierarchical Clustering (Agglomerative)", font_size=42)
        subtitle = Text("Average linkage • Euclidean distance", font_size=28)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(subtitle))

        axes = Axes(
            x_range=[1.0, 5.6, 1.0],
            y_range=[0.0, 2.2, 0.5],
            x_length=6.3,
            y_length=4.2,
            axis_config={"color": WHITE},
        )
        axes_labels = axes.get_axis_labels(
            x_label=Text("petal length (cm)", font_size=24),
            y_label=Text("petal width (cm)", font_size=24),
        )
        scatter_group = VGroup(axes, axes_labels)
        scatter_group.to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)

        # Dendrogram frame
        dendro_frame = Rectangle(width=6.2, height=4.5, stroke_color=WHITE, stroke_width=2)
        dendro_frame.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.35)
        dendro_title = Text("Dendrogram", font_size=28)
        dendro_title.next_to(dendro_frame, UP, buff=0.15)

        self.play(Create(axes), FadeIn(axes_labels), Create(dendro_frame), FadeIn(dendro_title))

        # ----------------------
        # Plot points with labels
        # ----------------------
        species_color = {
            "setosa": BLUE,
            "versicolor": GREEN,
            "virginica": ORANGE,
        }

        dots: Dict[int, Circle] = {}
        dot_labels: Dict[int, Text] = {}
        for p in pts:
            dot = Circle(radius=0.07, color=species_color.get(p.species, YELLOW), fill_opacity=1.0)
            dot.move_to(axes.c2p(p.x, p.y))
            label = Text(str(p.idx), font_size=22)
            label.next_to(dot, UP, buff=0.08)
            dots[p.idx] = dot
            dot_labels[p.idx] = label

        self.play(
            LaggedStart(
                *[FadeIn(dots[p.idx]) for p in pts],
                *[FadeIn(dot_labels[p.idx]) for p in pts],
                lag_ratio=0.06,
            )
        )

        legend = VGroup(
            VGroup(Circle(radius=0.06, color=BLUE, fill_opacity=1), Text("setosa", font_size=22)).arrange(RIGHT, buff=0.2),
            VGroup(Circle(radius=0.06, color=GREEN, fill_opacity=1), Text("versicolor", font_size=22)).arrange(RIGHT, buff=0.2),
            VGroup(Circle(radius=0.06, color=ORANGE, fill_opacity=1), Text("virginica", font_size=22)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.next_to(scatter_group, DOWN, buff=0.25).align_to(scatter_group, LEFT)
        self.play(FadeIn(legend))

        # ----------------------
        # Dendrogram coordinate system (manual)
        # ----------------------
        n = len(pts)
        # Leaf x positions (evenly spaced inside dendrogram frame)
        left = dendro_frame.get_left()[0]
        right = dendro_frame.get_right()[0]
        bottom = dendro_frame.get_bottom()[1]
        top = dendro_frame.get_top()[1]

        x_pad = 0.4
        y_pad = 0.35
        leaf_xs = np.linspace(left + x_pad, right - x_pad, n)
        leaf_y = bottom + y_pad

        # Scale dendrogram heights to available vertical space
        max_h = float(np.max(Z[:, 2]))
        def y_from_height(h: float) -> float:
            # map [0, max_h] -> [leaf_y, top-y_pad]
            return leaf_y + (top - y_pad - leaf_y) * (h / max_h if max_h > 0 else 0)

        # Leaf labels at the bottom
        leaf_texts: Dict[int, Text] = {}
        for i, p in enumerate(pts):
            t = Text(str(p.idx), font_size=22)
            t.move_to(np.array([leaf_xs[i], leaf_y - 0.20, 0]))
            leaf_texts[i] = t

        self.play(LaggedStart(*[FadeIn(t) for t in leaf_texts.values()], lag_ratio=0.05))

        # We will track each cluster's dendrogram node position (x, y)
        # Initial clusters: 0..n-1 correspond to original points in the same order as pts
        cluster_pos: Dict[int, Tuple[float, float]] = {i: (leaf_xs[i], leaf_y) for i in range(n)}
        cluster_members: Dict[int, List[int]] = {i: [pts[i].idx] for i in range(n)}

        # Visual groups for cluster highlighting in scatter
        current_hulls: Dict[int, Circle] = {}

        # Dendrogram lines drawn so far
        dendro_lines = VGroup()

        step_box = Rectangle(width=12.8, height=1.0, stroke_color=WHITE, stroke_width=2)
        step_box.next_to(subtitle, DOWN, buff=0.25)
        step_text = Text("Step 0: Start with each point as its own cluster", font_size=28)
        step_text.move_to(step_box.get_center())
        self.play(Create(step_box), FadeIn(step_text))

        # Helper to draw a soft circle around a cluster in scatter
        def highlight_cluster(cluster_id: int, color=YELLOW) -> Circle:
            members = cluster_members[cluster_id]
            # compute a centroid in scene coords
            pts_mobj = [dots[m].get_center() for m in members]
            centroid = np.mean(np.array(pts_mobj), axis=0)
            # radius = max distance to centroid + padding
            r = max(np.linalg.norm(p - centroid) for p in pts_mobj) + 0.25
            c = Circle(radius=r, color=color)
            c.move_to(centroid)
            c.set_stroke(width=6)
            return c

        # Animate each merge
        for step, (a, b, dist, new_size) in enumerate(Z, start=1):
            a = int(a)
            b = int(b)
            new_cluster_id = n - 1 + step  # SciPy uses implicit ids; we'll mirror with our own consistent ids

            # Update the step text
            new_step_text = Text(
                f"Step {step}: merge the closest clusters (distance = {dist:.2f})",
                font_size=28,
            )
            new_step_text.move_to(step_box.get_center())
            self.play(Transform(step_text, new_step_text))

            # Highlight the two clusters being merged
            ha = highlight_cluster(a, color=PURPLE)
            hb = highlight_cluster(b, color=RED)
            self.play(GrowFromCenter(ha), GrowFromCenter(hb))

            # Optional: indicate the points involved
            for m in cluster_members[a]:
                self.play(Indicate(dots[m], color=PURPLE), run_time=0.15)
            for m in cluster_members[b]:
                self.play(Indicate(dots[m], color=RED), run_time=0.15)

            # Build dendrogram branch for this merge
            xa, ya = cluster_pos[a]
            xb, yb = cluster_pos[b]
            y_new = y_from_height(float(dist))
            x_new = (xa + xb) / 2.0

            # Vertical lines up from children to new height
            va = Line(start=[xa, ya, 0], end=[xa, y_new, 0], color=WHITE)
            vb = Line(start=[xb, yb, 0], end=[xb, y_new, 0], color=WHITE)
            # Horizontal connector
            hline = Line(start=[xa, y_new, 0], end=[xb, y_new, 0], color=WHITE)

            # Height label
            h_label = Text(f"{dist:.2f}", font_size=20)
            h_label.next_to(hline, UP, buff=0.06)

            self.play(Create(va), Create(vb), Create(hline), FadeIn(h_label))
            dendro_lines.add(va, vb, hline, h_label)

            # Update bookkeeping for new cluster
            cluster_pos[new_cluster_id] = (x_new, y_new)
            cluster_members[new_cluster_id] = cluster_members[a] + cluster_members[b]

            # Remove temporary highlights
            self.play(FadeOut(ha), FadeOut(hb))

            # Set new cluster's base y for future vertical lines = y_new
            # (Children already stop at y_new, so next merge will start at y_new)
            # We do that by setting the cluster_pos for a and b to the new node too,
            # but keeping them separate would be confusing. Instead, we keep a/b as historical
            # and only use new_cluster_id from now on.

            # IMPORTANT: SciPy linkage refers to earlier formed clusters by increasing ids.
            # In Z, any a or b >= n refers to a previously created cluster. To mirror that,
            # we map SciPy cluster ids to our new_cluster_id scheme.
            # The simplest way: maintain an id-map from SciPy ids to our ids.

            # We'll implement that by updating a mapping after each step.

            # (We handle mapping outside the loop by rewriting Z indices before looping.)

        # End: show final emphasis
        final_text = Text("Complete hierarchy: cut the dendrogram at any height to choose k clusters", font_size=28)
        final_text.move_to(step_box.get_center())
        self.play(Transform(step_text, final_text))
        self.wait(2)


# --- Patch: SciPy cluster-id mapping ---
# The loop above assumes that Z refers to cluster ids that exist in our dictionaries.
# SciPy's linkage uses ids:
#   0..n-1 for original points
#   n..n+(n-2) for newly formed clusters
# We'll fix this by monkey-patching construct() to pre-map Z indices.

# This approach keeps the educational code readable while ensuring correct animation.

_original_construct = HierarchicalClusteringDemo.construct

def _construct_with_idmap(self: HierarchicalClusteringDemo):
    pts = load_points("data/iris_8.csv")
    X = np.array([[p.x, p.y] for p in pts], dtype=float)
    Z = linkage(X, method="average", metric="euclidean")

    # Build mapping from SciPy cluster ids -> our ids.
    # We'll keep original ids (0..n-1) as-is.
    n = len(pts)
    idmap: Dict[int, int] = {i: i for i in range(n)}
    next_id = n  # our first new cluster id will be n

    Z_mapped = []
    for i in range(Z.shape[0]):
        a, b, dist, size = Z[i]
        a = int(a); b = int(b)
        a_m = idmap[a]
        b_m = idmap[b]
        # assign new id for the newly formed cluster (which in SciPy is n+i)
        new_scipy_id = n + i
        new_our_id = next_id
        next_id += 1
        idmap[new_scipy_id] = new_our_id
        Z_mapped.append((a_m, b_m, float(dist), int(size)))

    # Store the mapped linkage on the instance and run a slightly modified construct
    self._Z_mapped = Z_mapped

    # Now run a local, fully correct version of the animation using the mapped linkage.
    self._construct_animation(pts, np.array(X), Z_mapped)


def _construct_animation(self: HierarchicalClusteringDemo, pts: List[PointRow], X: np.ndarray, Z_mapped: List[Tuple[int,int,float,int]]):
    # ----------------------
    # Layout: scatter (left) + dendrogram (right)
    # ----------------------
    title = Text("Hierarchical Clustering (Agglomerative)", font_size=42)
    subtitle = Text("Average linkage • Euclidean distance", font_size=28)
    title.to_edge(UP)
    subtitle.next_to(title, DOWN, buff=0.15)
    self.play(FadeIn(title), FadeIn(subtitle))

    axes = Axes(
        x_range=[1.0, 5.6, 1.0],
        y_range=[0.0, 2.2, 0.5],
        x_length=6.3,
        y_length=4.2,
        axis_config={"color": WHITE},
    )
    axes_labels = axes.get_axis_labels(
        x_label=Text("petal length (cm)", font_size=24),
        y_label=Text("petal width (cm)", font_size=24),
    )
    scatter_group = VGroup(axes, axes_labels)
    scatter_group.to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)

    dendro_frame = Rectangle(width=6.2, height=4.5, stroke_color=WHITE, stroke_width=2)
    dendro_frame.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.35)
    dendro_title = Text("Dendrogram", font_size=28)
    dendro_title.next_to(dendro_frame, UP, buff=0.15)

    self.play(Create(axes), FadeIn(axes_labels), Create(dendro_frame), FadeIn(dendro_title))

    # ----------------------
    # Plot points
    # ----------------------
    species_color = {
        "setosa": BLUE,
        "versicolor": GREEN,
        "virginica": ORANGE,
    }

    dots: Dict[int, Circle] = {}
    dot_labels: Dict[int, Text] = {}
    for p in pts:
        dot = Circle(radius=0.07, color=species_color.get(p.species, YELLOW), fill_opacity=1.0)
        dot.move_to(axes.c2p(p.x, p.y))
        label = Text(str(p.idx), font_size=22)
        label.next_to(dot, UP, buff=0.08)
        dots[p.idx] = dot
        dot_labels[p.idx] = label

    self.play(
        LaggedStart(
            *[FadeIn(dots[p.idx]) for p in pts],
            *[FadeIn(dot_labels[p.idx]) for p in pts],
            lag_ratio=0.06,
        )
    )

    legend = VGroup(
        VGroup(Circle(radius=0.06, color=BLUE, fill_opacity=1), Text("setosa", font_size=22)).arrange(RIGHT, buff=0.2),
        VGroup(Circle(radius=0.06, color=GREEN, fill_opacity=1), Text("versicolor", font_size=22)).arrange(RIGHT, buff=0.2),
        VGroup(Circle(radius=0.06, color=ORANGE, fill_opacity=1), Text("virginica", font_size=22)).arrange(RIGHT, buff=0.2),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    legend.next_to(scatter_group, DOWN, buff=0.25).align_to(scatter_group, LEFT)
    self.play(FadeIn(legend))

    # ----------------------
    # Dendrogram scaffold
    # ----------------------
    n = len(pts)
    left = dendro_frame.get_left()[0]
    right = dendro_frame.get_right()[0]
    bottom = dendro_frame.get_bottom()[1]
    top = dendro_frame.get_top()[1]

    x_pad = 0.4
    y_pad = 0.35
    leaf_xs = np.linspace(left + x_pad, right - x_pad, n)
    leaf_y = bottom + y_pad

    max_h = max(d for (_, _, d, _) in Z_mapped)
    def y_from_height(h: float) -> float:
        return leaf_y + (top - y_pad - leaf_y) * (h / max_h if max_h > 0 else 0)

    # Leaf labels
    for i, p in enumerate(pts):
        t = Text(str(p.idx), font_size=22)
        t.move_to(np.array([leaf_xs[i], leaf_y - 0.20, 0]))
        self.add(t)

    # Cluster bookkeeping for dendrogram geometry
    cluster_pos: Dict[int, Tuple[float, float]] = {i: (leaf_xs[i], leaf_y) for i in range(n)}
    cluster_members: Dict[int, List[int]] = {i: [pts[i].idx] for i in range(n)}

    # Step narration
    step_box = Rectangle(width=12.8, height=1.0, stroke_color=WHITE, stroke_width=2)
    step_box.next_to(subtitle, DOWN, buff=0.25)
    step_text = Text("Step 0: Start with each point as its own cluster", font_size=28)
    step_text.move_to(step_box.get_center())
    self.play(Create(step_box), FadeIn(step_text))

    def highlight_cluster(cluster_id: int, color=YELLOW) -> Circle:
        members = cluster_members[cluster_id]
        pts_mobj = [dots[m].get_center() for m in members]
        centroid = np.mean(np.array(pts_mobj), axis=0)
        r = max(np.linalg.norm(p - centroid) for p in pts_mobj) + 0.25
        c = Circle(radius=r, color=color)
        c.move_to(centroid)
        c.set_stroke(width=6)
        return c

    # Animate merges
    next_cluster_id = n
    for step, (a, b, dist, size) in enumerate(Z_mapped, start=1):
        new_id = next_cluster_id
        next_cluster_id += 1

        new_step_text = Text(
            f"Step {step}: merge closest clusters (avg distance = {dist:.2f})",
            font_size=28,
        ).move_to(step_box.get_center())
        self.play(Transform(step_text, new_step_text))

        ha = highlight_cluster(a, color=PURPLE)
        hb = highlight_cluster(b, color=RED)
        self.play(GrowFromCenter(ha), GrowFromCenter(hb))

        # draw dendrogram branch
        xa, ya = cluster_pos[a]
        xb, yb = cluster_pos[b]
        y_new = y_from_height(dist)
        x_new = (xa + xb) / 2.0

        va = Line(start=[xa, ya, 0], end=[xa, y_new, 0], color=WHITE)
        vb = Line(start=[xb, yb, 0], end=[xb, y_new, 0], color=WHITE)
        hline = Line(start=[xa, y_new, 0], end=[xb, y_new, 0], color=WHITE)
        h_label = Text(f"{dist:.2f}", font_size=20).next_to(hline, UP, buff=0.06)

        self.play(Create(va), Create(vb), Create(hline), FadeIn(h_label))

        # update clusters
        cluster_pos[new_id] = (x_new, y_new)
        cluster_members[new_id] = cluster_members[a] + cluster_members[b]

        # remove old clusters so they aren't used again
        # (not necessary, but keeps bookkeeping clean)
        cluster_pos.pop(a)
        cluster_pos.pop(b)
        cluster_members.pop(a)
        cluster_members.pop(b)

        self.play(FadeOut(ha), FadeOut(hb))

    final_text = Text("Done: the dendrogram records each merge and its height", font_size=28).move_to(step_box.get_center())
    self.play(Transform(step_text, final_text))
    self.wait(2)


# Bind the patched method
HierarchicalClusteringDemo.construct = _construct_with_idmap
HierarchicalClusteringDemo._construct_animation = _construct_animation
