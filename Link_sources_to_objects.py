"""
a script to link detected sources into objects.
Author: JJ Kavelaars
Date: 2022-08-05
File Name: Link_sources_to_objects.py
"""
# coding: utf-8
import pickle
import sys
import logging
import mp_ephem
import numpy
import pandas as pd
from astropy.time import Time
from matplotlib import pyplot
from scipy.optimize import minimize
import argparse


def ra_model(r0: float, rate: float, t0: float, t: numpy.ndarray, angle: float) -> numpy.ndarray:
    """
    Compute the ra of the sources based on the model of linear motion.
    Args:
        r0: zero-point of RA of this object
        rate: rate of motion of the object
        t0: zero-point of time this group of sources.
        t: array of times of observations of this source
        angle: angle of motion of source on the sky.
    Returns:
        (numpy.ndarray): array of model RA locations.
    """
    return r0 + rate * (24. / 3600) * (t-t0) * numpy.cos(numpy.radians(angle))


def dec_model(d0: float, rate: float, t0: float, t: numpy.ndarray, angle: float) -> numpy.ndarray:
    """
    Compute the dec of the sources based on the model of linear motion.
        Args:
        d0: zero-point of RA of this object
        rate: rate of motion of the object
        t0: zero-point of time this group of sources.
        t: array of times of observations of this source
        angle: angle of motion of source on the sky.
    Returns:
        (numpy.ndarray): array of model Dec locations.
    """
    return d0 + rate * (24. / 3600) * (t-t0) * numpy.sin(numpy.radians(angle))


def lnp(theta: list, r: numpy.ndarray, d: numpy.ndarray, t: numpy.ndarray,
        ra_err: float = 0.1875 / 3600, dec_err: float = 0.1875 / 3600) -> float:
    """
    Args:
        theta: array of arguments to function to be fit r0, d0, t0 rate, angle (intercepts of the line and slope)
        r: the RA position of the source
        d: the Dec position of the source
        t: the Time of the observation
        ra_err: uncertainty weight of the RA position
        dec_err: uncertainty weight of the Dec position

    Returns:
        float: the linear sum of the ra and dec residuals

    This function computes the residuals between the current line parameters and measured positions of sources.
    """
    r0, d0, t0, rate, angle = theta
    ra_residuals = numpy.sum(((r - ra_model(r0, rate, t0, t, angle)) ** 2) / (ra_err ** 2))
    dec_residuals = numpy.sum(((d - dec_model(d0, rate, t0, t, angle)) ** 2) / (dec_err ** 2))
    return ra_residuals + dec_residuals


# maximum-likelihood estimation part
def fit_rate_line(ra_obs: numpy.ndarray, dec_obs: numpy.ndarray, time_obs: numpy.ndarray,
                  rate_init: float = -3.0, angle_init: float = 10.0) -> list:
    """
    Args:
        ra_obs: array of RA positions of the source
        dec_obs: array of Dec positions of the source
        time_obs: array of the Time of observation (in JD)
        rate_init: initial guess for rate of motion
        angle_init: initial guess for angle of motion

    Returns:
        (list): maximum likelihood solution object (from minimize) and array indicating
        which observations are within 4 sigma of the line
    """
    numpy.random.seed(0)
    ra_err = 0.1875/3600.
    dec_err = ra_err
    initial = numpy.array([ra_obs[0], dec_obs[0], time_obs[0], rate_init, angle_init])
    solution = minimize(lnp, initial, args=(ra_obs, dec_obs, time_obs))

    # compute an array that holds 'True' for all points that are within 4-sigma of the best-fit line.
    condition_ra = ((ra_obs-ra_model(solution.x[0], solution.x[3], solution.x[2], time_obs, solution.x[4]))**2 / ra_err**2) < 4
    condition_dec = ((dec_obs-dec_model(solution.x[1], solution.x[3], solution.x[2], time_obs, solution.x[4]))**2 / dec_err**2) < 4
    condition = condition_ra & condition_dec

    # recompute the solution with only those points that are within 4-sigma of initial fit.
    initial = solution.x
    solution = minimize(lnp, initial, args=(ra_obs[condition], dec_obs[condition], time_obs[condition]))

    # return the solution object and a array showing which points are within 4-sigma of the best fit line.
    return solution.x


def make_observation(df: pd.DataFrame, band: str = 'r', observatory_code: int = 568) -> list:
    """
    Given a panda dataframe in the format the comes from the detection process create a list of
    minor-planet-centre formatted observations that can be passed to orbit fitting codes and reported to MPC

    Args:
         df: A panda DataFrame that contains the positions and time of observations of a
         group of sources thought to be the same object
         band: pass-band of the observations
         observatory_code: MinorPlanetCenter observatory code  568 -> Maunakea
    """
    obs = []
    for index in df.index:
        ra = df['ra'][index]
        dec = df['dec'][index]
        mag = df['pred_mag']

        # makeup a name for the source, here we put an 'M" in front of the index number of the first source in the object
        provisional_name = f"M{df.index[0]}"
        date = Time(df['time'][index], format='mjd', precision=5)

        # leave the bandpass blank, but could be passes
        band = band
        observatory_code = observatory_code

        # we also have extended information, beyond the normal MPC values to indicate the exposure number and location of measurement.
        frame = f"df['expnum0'][index]"
        x_position = df['pred_pos_x0'][index]
        y_position = df['pred_pos_y0'][index]
        obs.append(mp_ephem.Observation(provisional_name=provisional_name,
                                        ra=ra,
                                        dec=dec,
                                        mag=mag,
                                        date=date,
                                        band=band,
                                        observatory_code=observatory_code,
                                        frame=frame,
                                        xpos=x_position,
                                        ypos=y_position))
    return obs
        

def build_objects(df: pd.DataFrame, rate_min: float = -150., rate_max: float = -15, angle_min = -1.5, angle_max = 15, d_mag_max: float = 0.5) -> list:
    """
    Build a list of sources that may be grouped as objects based on proximity on the sky.  Each list entry is a list of the sources
    that could be the same object.

    Args:
        df: the panda DataFrame that comes from the detection pipeline
        rate_min: the minimum rate of motion that an object could have, sets size of area that links sources together.
        rate_max: the maximum rate of motion that an object could have, sets size of area that links sources together.
        Change those rates to find faster moving objects. (I used from -15 to -30 to find some SSOs)
        d_mag_max: maximum magnitude difference between different sources that make-up the same object.

    Returns:
         (list): a list of lists of sources that make up objects.
    """

    # to define angle difference
    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    
    # hits contains the list of linked sources.
    hits = []

    # create a new column in the DataFrame that indicates if a sources has been matched into an object already.
    df['matched'] = False

    logging.info(f"Checking {len(df)} possible object links\n")

    # loop over the entire data frame.
    for index in range(len(df)):
        # if this sources is already linked to an object, then skip
        
        if index % 100 == 0:
            logging.debug(f"Using source at index {index:5d}/{len(df)} as anchor point for object creation\n")
        
        if df['matched'][index]:
            continue

        # get an array that indicates which sources to try and make a connection with.
        # don't match the currently selected anchor source and don't select any sources that are already matched.
        cases = (df['expnum0'] != df['expnum0'][index]) & (~df['matched'])

        # now compute the rate of motion between all the sources that cases allows and the anchor source.
        rate = 3600.0*numpy.sqrt(((df['ra'][cases] - df['ra'][index])*numpy.cos(numpy.radians(df['dec'][index])))**2
                                 + (df['dec'][cases] - df['dec'][index])**2)/(24*(df['time'][cases]-df['time'][index]))
        
        # also have a condition on magnitude of the source.
        mag = numpy.fabs(df['pred_mag'][cases] - df['pred_mag'][index]) < d_mag_max
        # Members now contains all the sources that could match with the anchor source to be an object.
        members = (cases & (rate > rate_min) & (rate < rate_max) & mag) | (df.index == index)
        # members = (cases & (rate > rate_min) & (rate < rate_max)) | (df.index == index)
        
        if members.sum() < 3:
            # too few hits to make an object.
            continue

        # obs now contains a list of all sources that could be part of this object based on the starting/anchor source.
        obs = df.loc[members]
        indexes = obs.index.values

        # fit a line to these sources to see if they form a linear structure.
        ra_obs = numpy.array(obs['ra'])
        dec_obs = numpy.array(obs['dec'])
        time_obs = numpy.array(obs['time'])
        solution = fit_rate_line(ra_obs, dec_obs, time_obs)

        # compute the residuals in arc-seconds between the observed values and the best-fit line
        x_obs = 3600.0*(ra_obs-solution[0])   # ra of observation
        y_obs = 3600.0*(dec_obs-solution[1])  # dec of observations
        x_mod = (ra_model(solution[0], solution[3], solution[2], time_obs, solution[4])-solution[0])*3600.0  # ra from best-fit line
        y_mod = (dec_model(solution[1], solution[3], solution[2], time_obs, solution[4])-solution[1])*3600.0  # dec from best-fit line
        ra_residuals = (x_obs - x_mod)  # ra residuals
        dec_residuals = (y_obs - y_mod)  # dec residuals

        # remove sources that are more then 0.5 arc-seconds from the best fit line.
        cond = (numpy.fabs(ra_residuals) < 0.5) & (numpy.fabs(dec_residuals) < 0.5)
        indexes = indexes[cond]  # now index only has the source that are within 0.5 arc-seconds of best-fit line in ra and dec

        if cond.sum() < 3:
            # too few hits to make an object.
            continue

        # get rid of the duplicate sources in the list of observations,
        # the same source can appear multiple times in the detection DataFrame
        # duplicates are identified by the exposure number attribute 'expnum0'
        try:
            obs = obs.loc[cond].drop_duplicates(subset='expnum0')
            # if this reduces to less than 3 observations then reject, as we need 3 observations to get a detections.
            if len(obs) < 3:
                continue
            residual_condition = numpy.ones(len(obs), dtype=bool)
        except Exception as ex:
            logging.error(f"exception: {ex}")
            continue
        if residual_condition.sum() < 3:
            sys.stderr.write(f"{index:5d}: Too few observations survive orbit fitting residual cut.\n")
            continue

        # add the list of sources that form this object to the 'hits' list, which is a list of lists.
        hits.append(obs.loc[residual_condition].sort_values(by='expnum0'))

        # set the 'matched' column of the original DataFrame based on sources that have been connected to this object
        # using the index values from those observations that have been added to the current object.
        not_matched = numpy.array(obs.index.tolist())[~residual_condition]

        for matched_index in indexes:
            df.loc[matched_index, 'matched'] = True
        for not_matched_index in not_matched:
            df.loc[not_matched_index, 'matched'] = False

    return hits


def link_hits(hits: list) -> list:
    """
    Given a list of 'hits' (from build_objects) determine if there are multiple hits that are the same object.

    Args:
        hits: a list of lists of sources that are tentatively formed into objects.
    Returns:
        (list): a list of linked sources after grouping objects from the hits list.
    """
    links = []
    logging.info(f"Matching {len(hits)} objects\n")
    for idx1 in range(len(hits)):
        logging.debug(f"Looking at entry {idx1:5d}\r")
        # get the list of observations from this tentative object.
        obs = hits[idx1]
        ra_obs = numpy.array(obs['ra'])
        dec_obs = numpy.array(obs['dec'])
        time_obs = numpy.array(obs['time'])
        # fit the line again
        solution = fit_rate_line(ra_obs, dec_obs, time_obs)
        # consider that these sources are all matched to form an object. The matched attribute starts as False for everyone as set
        # in the build_objects... in build_objects only copies of the DataFrame has matched set to true.
        hits[idx1]['matched'] = True
        # loop over the remaining objects to see if any are the same object as this one.
        for idx2 in range(idx1+1, len(hits)):
            # if all these sources are already matched to an object then skip this one.
            if hits[idx2]['matched'][hits[idx2].index[0]]:
                continue
            # get all the source observations for this object
            obs = hits[idx2]
            ra_obs = numpy.array(obs['ra'])
            dec_obs = numpy.array(obs['dec'])
            time_obs = numpy.array(obs['time'])

            # compute the residuals between this set of sources and the line fit to the first set of sources.
            x_obs = 3600.0*(ra_obs-solution[0])
            y_obs = 3600.0*(dec_obs-solution[1])
            x_mod = (ra_model(solution[0], solution[3], solution[2], time_obs, solution[4])-solution[0])*3600.0
            y_mod = (dec_model(solution[1], solution[3], solution[2], time_obs, solution[4])-solution[1])*3600.0
            ra_residuals = (x_obs - x_mod)
            dec_residuals = (y_obs - y_mod)

            # only keep these sources if they are within 0.5 arc-seconds of the line from the first set of sources.
            # require that at least 3 of the measures from this object could be observations of the exterior loop objet.
            cond = (numpy.fabs(ra_residuals) < 0.5) & (numpy.fabs(dec_residuals) < 0.5)
            if cond.sum() < 3:
                continue
            # looks like there are a good match, so append them to the linked list
            links.append([idx1, idx2])
            # mark this object has having been matched... so we skip it later.
            hits[idx2]['matched'] = True
    return links


def build_linked_objects(links: list, hits: list):
    """
    Take the links list and gather the observations from hits list that could be part of this sources based on original linking.

    Args:
        links: a list of lists of indexes that group sources in hits into objects.
        hits: a list of lists of sources that form objects.
    """
    # turn the links list into a numpy array.
    links_array = numpy.array(links)
    linked_hits = []
    # transpose the array so we can loop over the first sources in each object, uniquely.
    for idx1 in numpy.unique(links_array.T[0]):
        linked = hits[idx1]
        # go through the linked list to get all the entries that also hold this source, ie the same object.
        # and append those source measurements.
        for idx2 in links_array[links_array.T[0] == idx1].T[1]:
            linked = linked.append(hits[idx2])
        # remove duplicate observations.
        linked = linked.drop_duplicates(subset='expnum0')
        # linked_hist is our list of lists of sources that have been linked into the same object.
        linked_hits.append(linked.sort_values(by=['expnum0']))
    # Add objects that didn't have multiple detections
    multi_hits = numpy.append(numpy.unique(links_array.T[0]), numpy.unique(links_array.T[1]))
    for idx in range(len(hits)):
        if idx not in multi_hits:
            if len(hits[idx]) < 2:
                continue
            linked_hits.append(hits[idx])
    return linked_hits


def diagnostic(linked_hits: list) -> None:
    """
    Make plots for each object to see how the sources match to a line... a good diagnostic step
    Args:
        linked_hits: a list of lists of observations that link to form sources.
    """

    for i, obs in enumerate(linked_hits):
        pyplot.clf()
        ra_obs = numpy.array(obs['ra'])
        dec_obs = numpy.array(obs['dec'])
        time_obs = numpy.array(obs['time'])
        solution = fit_rate_line(ra_obs, dec_obs, time_obs)
        x_obs = 3600.0 * (ra_obs - solution[0])
        y_obs = 3600.0 * (dec_obs - solution[1])
        x_mod = (ra_model(solution[0], solution[3], solution[2], time_obs, solution[4]) - solution[0]) * 3600.0
        y_mod = (dec_model(solution[1], solution[3], solution[2], time_obs, solution[4]) - solution[1]) * 3600.0
        ra_residuals = (x_obs - x_mod)
        dec_residuals = (y_obs - y_mod)
        cond = (numpy.fabs(ra_residuals) < 0.5) & (numpy.fabs(dec_residuals) < 0.5)
        pyplot.plot(x_obs[cond], y_obs[cond], 'o')
        pyplot.plot(x_mod[cond], y_mod[cond], 'x')
        xx = numpy.vstack([x_obs, x_mod])
        yy = numpy.vstack([y_obs, y_mod])
        pyplot.plot(xx, yy, '--k')
        pyplot.show()
        pyplot.savefig(f'diagnostic_{i:02d}.png')
        input("Press Enter to continue...")


def main(csv_file: str) -> list:
    """
    Do the linking between sources to find objects.

    Args:
        csv_file: a CSV formatted dump of the detected sources panda.DataFrame.
    """
    df = pd.read_csv(csv_file)
    hits = build_objects(df)
    links = link_hits(hits)
    linked_hits = build_linked_objects(links, hits)

    lens = [len(linked_hits[idx]) for idx in range(len(linked_hits))]
    pyplot.hist(lens, bins=range(3, 44))
    cond = (numpy.array(lens) > 2)
    good_links = [linked_hits[x] for x in numpy.arange(len(linked_hits))[cond]]
    print(good_links)
    numpy.save(f'{csv_file[:-4]}_linked_hits.npy', good_links)
    return linked_hits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help="CSV file holding the Pandas Dataframe of detected sources.")
    parser.add_argument('--show-plots', '--p', action='store_true', help="Show diagnostic line plot of detected objects")
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose outputs')
    parser.add_argument('--debug', '--d', action='store_true')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)
    linked_list = main(args.csv_file)
    if args.show_plots:
        diagnostic(linked_list)