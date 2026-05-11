FLEET MONITORING – FAST DASHBOARD
==================================

WHAT IS THIS?

A Dash dashboard that displays precomputed KPI results instantly, without loading any raw data.
All results are pre-generated offline and stored as Parquet files. The dashboard just reads
those files and renders charts and tables.


ARCHITECTURE

  generate_all_precomputed.py   Run this once offline to compute all KPI combinations.
  fast_dashboard_reader.py      The dashboard itself (reads only Parquet, never raw data).
  data/                         Folder containing all precomputed results.
    view1/kpis/                   KPI 1 to 6 results
    view1/kpi7/                   KPI 7 results
    view2/kpi8/                   KPI 8 results

Workflow: run the generator once, then launch the dashboard. No computation happens at runtime.


==================================
VUE 1 – MAIN KPIs + FUEL EVOLUTION
==================================

DATE LOGIC

The main calculation date is the COB date (Close of Business date), which represents the
snapshot date of the fleet — i.e. the state of the fleet on a given month-end. By default,
all KPIs are computed on the stock present at COB date.

For KPI 1 to 6 and KPI 8, you can optionally choose to filter vehicles using a different date
within that same COB month, instead of the COB date itself:

  NONE               Default. Uses the COB date — shows the full fleet stock at that date.
  CONTRACT_START_DATE  Only includes vehicles whose contract started in the COB month.
  DELIVERY_DATE        Only includes vehicles that were delivered in the COB month.

This lets you analyse either the overall stock (NONE) or the flow of new contracts or
deliveries within a given period.


KPI 1   Share of contracts with duration under 25 months.
KPI 2   Share of contracts with duration between 25 and 30 months.
KPI 3   Diesel vs non-diesel split (%).
KPI 4   Hybrid vehicle share (%).
KPI 5   Electric vehicle (EV) share (%).
KPI 6   Passenger car (PV) vs light commercial vehicle (LCV) split (%).

Filters for KPI 1 to 6:
A global filter bar at the top sets all filters for all 6 cards at once. Each card also has
its own individual filters that override the global ones — useful to compare two cards with
different settings side by side. Click Refresh to apply.

  Country         The country to analyse, or ALL for the full portfolio.
  Year            The reference year.
  Month           A specific month, or ALL for the full year.
  Asset Status    IN FLEET (active vehicles), ORDER (vehicles on order), DEHIRE (vehicles
                  being returned), SOLD, or ALL to include everything.
  Date Rule       See date logic above: NONE, CONTRACT_START_DATE, or DELIVERY_DATE.
  Vehicle Type    CAR, BIKE, or ALL.

The global filter overwrites the 6 individual card filters when applied. Each card filter can
then be changed individually afterwards to override the global value for that card only.


KPI 7 – Fuel type evolution:
Shows how the fuel type mix (diesel, petrol, hybrid, EV...) evolves over time, as a share or
volume. KPI 7 always uses the COB date as its reference date — there is no date rule option
here. Its filters are completely independent from the KPI 1-6 bar. Click Refresh KPI 7 to apply.

  Country         The country to analyse, or ALL.
  Date range      Start and end month for the evolution chart.
  Asset Status    IN FLEET, ORDER, DEHIRE, SOLD, or ALL.
  Period          Monthly, quarterly, or yearly grouping.
  Vehicle Type    CAR, BIKE, or ALL.
  View            Share (%) or Volume (number of vehicles).


==================================
VUE 2 – PRODUCTION BY ENERGY
==================================

KPI 8 – Production by fuel type:
Shows new vehicle production broken down by fuel type, month by month over the year. The date
rule applies here the same way as for KPI 1-6: you can choose whether to count vehicles by
COB date, contract start date, or delivery date within each month. Asset status is fixed to
IN FLEET for this view. Click Refresh to apply.

  Country         The country to analyse, or ALL.
  Year            The reference year.
  Asset Status    Fixed to IN FLEET.
  Date Rule       NONE, CONTRACT_START_DATE, or DELIVERY_DATE.
  Vehicle Type    CAR, BIKE, or ALL.
  Metric          Share (%) or Volume (number of vehicles).
  Period          Monthly, quarterly, or yearly grouping.
