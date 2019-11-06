import numpy as np
import matplotlib.pyplot as plt

class wind_farm_plot():

    def __init__(self, turb_x, turb_y, turb_diam, bound_x=np.array([0]), bound_y=np.array([0])):

        """

        Parameters
        ----------
        turb_x          1d numpy array of x coordinates of wind turbines
        turb_y          1d numpy array of y coordinates of wind turbines
        turb_diam       1d numpy array of diameter of wind turbine rotors
        bound_x         1d numpy array of x coordinates of boundary vertices, or center and one edge point for circle
        bound_y         1d numpy array of y coordinates of boundary vertices, or center and one edge point for circle
        Returns         axis with turbine locations and boundaries
        -------

        """

        self.turb_x = turb_x
        self.turb_y = turb_y
        self.turb_diam = turb_diam
        self.nTurbines = turb_x.size

        self.bound_x = bound_x
        self.bound_y = bound_y
        self.nVertices = bound_x.size

        # formatting
        self.bound_line_color = 'k'
        self.bound_line_type = '--'
        self.turb_line_color = 'r'
        self.turb_line_type = '-'

        # labels
        self.bound_label = 'Boundary'
        self.turb_label = 'Turbines'
        self.xlabel = 'Turbine X Position ($X/D_r$)'
        self.ylabel = 'Turbine Y Position ($Y/D_r$)'

    def plot_wind_farm(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        self._add_wind_farm_boundary()

        self._add_wind_turbine_locations()

        self.ax.axis('equal')

        self.legend = self.ax.legend(frameon=False)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        # self.fig.legend()

        return 0

    def save_wind_farm(self, filename):

        self.fig.savefig(filename, transparent=True)

    def show_wind_farm(self):
        plt.sca(self.ax)
        plt.show()

    def _add_wind_farm_boundary(self):

        # if too few vertices for a boundary, don't make a boundary
        if self.nVertices < 2:
            return

        # if two vertices, then the boundary is a circle and the second verticy is the radius
        elif self.nVertices == 2:
            boundary_center_x = self.bound_x[0]
            boundary_center_y = self.bound_y[0]
            boundary_radius = self.bound_x[1]

            boundary_circle = plt.Circle((boundary_center_x / self.turb_diam, boundary_center_y / self.turb_diam),
                                         boundary_radius / self.turb_diam, facecolor='none', edgecolor=self.bound_line_color,
                                         linestyle=self.bound_line_type, label=self.bound_label)

            self.ax.add_patch(boundary_circle)

        # if more than 2 vertices, then the boundary is a polygon
        else:

            bounds = np.column_stack([self.bound_x, self.bound_y])

            boundary_polygon = plt.Polygon(bounds/self.turb_diam, closed=True, facecolor="none", edgecolor=self.bound_line_color,
                                           linestyle=self.bound_line_type, label=self.bound_label)

            self.ax.add_patch(boundary_polygon)

        return 0

    def _add_wind_turbine_locations(self):
        i = 0
        for x, y in zip(self.turb_x / self.turb_diam, self.turb_y / self.turb_diam):
            i += 1
            if i == self.nTurbines:
                circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor=self.turb_line_color,
                                          linestyle=self.turb_line_type, label=self.turb_label)
            else:
                circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor=self.turb_line_color,
                                          linestyle=self.turb_line_type)

            self.ax.add_patch(circle_start)


        return 0


if __name__ ==  "__main__":

    turbx = np.array([0, 100, 200])
    turby = np.array([0, 100, 200])
    boundx = np.array([0, 300, 400, 500])
    boundy = np.array([0, 400, 400, 0])
    turb_diam = 80.
    myFarmPlot = wind_farm_plot(turbx, turby, turb_diam, boundx, boundy)

    myFarmPlot.plot_wind_farm()
    myFarmPlot.show_wind_farm()
    myFarmPlot.save_wind_farm('testing.pdf')

