include "alldifferent.mzn";
include "strictly_increasing.mzn";
include "increasing.mzn";
include "gecode.mzn";


%%%%%%%%%%%%%
%% INPUTS:
%%%%%%%%%%%%%


% Anzahl Spieler:innen
int: n_players;

% Anzahl Spiele
int: n_matchups;

array[1..n_players] of float: ELO;

array[1..n_players,1..n_players] of 0..256: played_together_cost;       % haben bereits gegeneinander gespielt
array[1..n_players,1..n_players] of 0..256: played_together_cost_team;  % haben bereits miteinander gespielt


float: higher_rating_weight;


%%%%%%%%%%%%%
%% Hilfsvariablen:
%%%%%%%%%%%%%

% 1D = 1dimensional -> es ist egal wer von den 4 Personen zusammenspielt
% wie beim gesamt Elo Gap, welches immer die Differenz zwischen Person 1 und 4 beträgt
% Bei den 2D Costs spielt es eine Rolle wie sich die Teams zusammensetzen. Alle 3 Möglichkeiten werden berechnet und mit arg_min wird die beste ausgewählt.

enum COSTS_1D = {ELO_GAP};
enum COSTS_2D = {TEAM_DIFF, PLAYED_TOGETHER};

% Konstante Faktoren mit denen die drei Kostenpunkte verrechnet werden
array[COSTS_1D] of float: cost_factor_1d;
array[COSTS_2D] of float: cost_factor_2d;

array[COSTS_1D] of float: reduce_thresh_1d;
array[COSTS_2D] of float: reduce_thresh_2d;

% für jedes Matchup die Kosten, jeweils 1D und 2D getrennt
array[1..n_matchups, COSTS_1D] of var 0.0..3000.0: cost_1d_raw;
array[1..n_matchups, COSTS_2D, 1..3] of var 0.0..3000.0: cost_2d_raw;

% Adjusted costs: Bei ELO_GAP und TEAM_DIFF gibt es einen treshold unter welchem die Kosten mit einem Faktor unter 0 multipliziert werden.
% Zu einem gewissen Maß müssen diese beiden Werte (pro matchup) ja über 0 sein und bis zu einem bestimmten Bereich ist das ja auch okay.
array[1..n_matchups, COSTS_1D] of var 0.0..90000.0: cost_1d_adjusted;
array[1..n_matchups, COSTS_2D, 1..3] of var 0.0..90000.0: cost_2d_adjusted;

% die Summe der 2D Kosten für alle 3 möglichen Konstellationen
array[1..n_matchups, 1..3] of var 0.0..90000.0: cost_2d_pos_sum;

% die minimalen Kosten - also die der besten Konstellation
array[1..n_matchups, COSTS_2D] of var 0.0..90000.0: cost_2d_min;


% Die played together Kosten für jedes Team in jeder Konstellation (pro Matchup)
array[1..n_matchups,1..2,1..3] of var 0.0..90000.0: matchup_table_team_costs;

%% Das Hertzstück: Die Tabelle mit den Spieler:innen Indicies
array[1..n_matchups,1..4] of var 1..n_players: matchup_table :: random;

array[1..n_matchups,1..2,1..3] of var 0.0..3000.0: matchup_table_team_elos;
array[1..n_matchups] of var 0.0..3000.0: matchup_table_elo_gap;

% Welche Teamkonstellation in den Matchups gespielt werden soll
array[1..n_matchups] of var 1..3: matchup_order;

array[1..3,1..4] of int: POS = [|1,4, 2,3,
                                |1,3, 2,4,
                                |1,2, 3,4|];



constraint (forall (mi in 1..n_matchups) (
  cost_2d_pos_sum[mi,..] = [sum(cost_2d_adjusted[mi,..,1]),
                                 sum(cost_2d_adjusted[mi,..,2]),
                                 sum(cost_2d_adjusted[mi,..,3])]));

constraint (forall (mi in 1..n_matchups) 
  (argmin_le(matchup_order[mi], cost_2d_pos_sum[mi,..] )));

        

%% ORDER 1 = 14 23
%% ORDER 2 = 13 24
%% ORDER 3 = 12 34

%% 1D RAW
constraint forall (mi in 1..n_matchups) (
      cost_1d_raw[mi,ELO_GAP] = ELO[matchup_table[mi,4]] - ELO[matchup_table[mi,1]]);

% constraint forall (mi in 1..n_matchups) (
%       cost_1d_raw[mi,ELO_GAP] = max([ELO[x] | x in matchup_table[mi,..]]) - min([ELO[x] | x in matchup_table[mi,..]]));


%% 1D ADJUSTED
constraint forall (mi in 1..n_matchups) (
      cost_1d_adjusted[mi,ELO_GAP] = cost_factor_1d[ELO_GAP] * 
      square(reduce_if_below(
          cost_1d_raw[mi,ELO_GAP],
          reduce_thresh_1d[ELO_GAP])));

%% 2D RAW
constraint forall (mi in 1..n_matchups) (
      forall (pos in 1..3) (
        cost_2d_raw[mi,TEAM_DIFF,pos] = abs(matchup_table_team_elos[mi,1,pos] - matchup_table_team_elos[mi,2,pos])
     ));


      
constraint forall (mi in 1..n_matchups) (
      forall (pos in 1..3) (
        matchup_table_team_costs[mi,1,pos] = played_together_cost_team[matchup_table[mi,POS[pos,1]],matchup_table[mi,POS[pos,2]]]
    /\  matchup_table_team_costs[mi,2,pos] = played_together_cost_team[matchup_table[mi,POS[pos,3]],matchup_table[mi,POS[pos,4]]]
    
%         cost_2d_raw[mi,PLAYED_TOGETHER,pos] =
%   played_together_cost_team[matchup_table[mi,POS[pos,1]],matchup_table[mi,POS[pos,2]]] +
%   played_together_cost_team[matchup_table[mi,POS[pos,3]],matchup_table[mi,POS[pos,4]]] %+
  
%   played_together_cost[matchup_table[mi,POS[pos,1]], matchup_table[mi,POS[pos,3]]] + 
%   played_together_cost[matchup_table[mi,POS[pos,1]], matchup_table[mi,POS[pos,4]]] + 
%   played_together_cost[matchup_table[mi,POS[pos,2]], matchup_table[mi,POS[pos,3]]] + 
%   played_together_cost[matchup_table[mi,POS[pos,2]], matchup_table[mi,POS[pos,4]]]
  ));

constraint forall (mi in 1..n_matchups) (
      forall (pos in 1..3) (
        cost_2d_raw[mi,PLAYED_TOGETHER,pos] = 
        matchup_table_team_costs[mi,1,pos] + matchup_table_team_costs[mi,2,pos] +
        
        played_together_cost[matchup_table[mi,POS[pos,1]], matchup_table[mi,POS[pos,3]]] + 
        played_together_cost[matchup_table[mi,POS[pos,1]], matchup_table[mi,POS[pos,4]]] + 
        played_together_cost[matchup_table[mi,POS[pos,2]], matchup_table[mi,POS[pos,3]]] + 
        played_together_cost[matchup_table[mi,POS[pos,2]], matchup_table[mi,POS[pos,4]]]
        ));
      
     
%% 2D ADJUSTED
constraint forall (mi in 1..n_matchups) (
      forall (pos in 1..3) (
      cost_2d_adjusted[mi,TEAM_DIFF,pos] = cost_factor_2d[TEAM_DIFF] * 
      square(reduce_if_below(
          cost_2d_raw[mi,TEAM_DIFF,pos],
          reduce_thresh_2d[TEAM_DIFF]))));
      
constraint forall (mi in 1..n_matchups) (
      forall (pos in 1..3) (
      cost_2d_adjusted[mi,PLAYED_TOGETHER,pos] = cost_factor_2d[PLAYED_TOGETHER] * 
      square(
        reduce_if_below(
          cost_2d_raw[mi,PLAYED_TOGETHER,pos],
          reduce_thresh_2d[PLAYED_TOGETHER]))));
      


%% 2D MIN
constraint forall (mi in 1..n_matchups) (
      cost_2d_min[mi,TEAM_DIFF] = cost_2d_adjusted[mi,TEAM_DIFF,matchup_order[mi]]);

constraint forall (mi in 1..n_matchups) (
      cost_2d_min[mi,PLAYED_TOGETHER] = cost_2d_adjusted[mi,PLAYED_TOGETHER,matchup_order[mi]]);
    

constraint alldifferent(matchup_table);

% constraint symmetry_breaking_constraint (forall (match_i in 1..n_matchups-1) 
% (     matchup_table[match_i,1+1] > matchup_table[match_i,1]+1 ) );


% function var float: played_together_cost_single_line(array[int] of var int: matchup) = 
%   played_together_cost_team[matchup[1],matchup[2]] + played_together_cost_team[matchup[3],matchup[4]] +
%   played_together_cost[matchup[1], matchup[3]] + played_together_cost[matchup[1], matchup[4]] + 
%   played_together_cost[matchup[2], matchup[3]] + played_together_cost[matchup[2], matchup[4]];
  

var float: COST_ELO_DIFF = sum(cost_2d_min[..,TEAM_DIFF]);
var float: COST_ELO_GAP =  sum(cost_1d_adjusted[..,ELO_GAP]);
var float: COST_PLAYED_TOGETHER = sum(cost_2d_min[..,PLAYED_TOGETHER]);

var float: COST;

constraint (
    COST =   sum(cost_1d_adjusted[..,ELO_GAP]) + 
             sum(cost_2d_min[..,TEAM_DIFF]) +
             sum(cost_2d_min[..,PLAYED_TOGETHER])
             );

%int: RESTART_CONSTANT = 750;

solve :: restart_constant(1250)
  minimize COST;

output ["M\(mi): \(matchup_table[mi,..])\t[\(matchup_order[mi])]\tTEAM-D: \(cost_2d_raw[mi,TEAM_DIFF,matchup_order[mi]]) ELO-GAP:\(cost_1d_raw[mi,ELO_GAP]) PLAYED-T: \(cost_2d_raw[mi,PLAYED_TOGETHER,matchup_order[mi]])  \([ELO[i] | i in matchup_table[mi,..]]) TEAM_DIFF:\(cost_2d_adjusted[mi,TEAM_DIFF,..]) PLAYED_T:\(cost_2d_adjusted[mi,PLAYED_TOGETHER,..])\n"  | mi in 1..n_matchups      ] ++ [show(COST_ELO_DIFF), " + ", show(COST_ELO_GAP), " + ", show(COST_PLAYED_TOGETHER), " = ", show(COST)]
%   ++ ["\n", show(played_together_cost_team)]
  ;




%% RATHER BORING STUFF:


% all rows have descending elos - why not ids?? 
% constraint (forall (match_i in 1..n_matchups) (
%       ELO[matchup_table[match_i,1]] >= ELO[matchup_table[match_i,2]]
%   /\  ELO[matchup_table[match_i,1]] >= ELO[matchup_table[match_i,3]]  
%   /\  ELO[matchup_table[match_i,1]] >= ELO[matchup_table[match_i,4]]
%   /\  ELO[matchup_table[match_i,2]] >= ELO[matchup_table[match_i,3]]
%   /\  ELO[matchup_table[match_i,2]] >= ELO[matchup_table[match_i,4]]
%   /\  ELO[matchup_table[match_i,3]] >= ELO[matchup_table[match_i,4]] ) );
constraint (forall (match_i in 1..n_matchups) (
      matchup_table[match_i,1] > matchup_table[match_i,2]
  /\  matchup_table[match_i,1] > matchup_table[match_i,3]
  /\  matchup_table[match_i,1] > matchup_table[match_i,4]
  /\  matchup_table[match_i,2] > matchup_table[match_i,3]
  /\  matchup_table[match_i,2] > matchup_table[match_i,4]
  /\  matchup_table[match_i,3] > matchup_table[match_i,4] ) );


% the team elos for all three permutations
constraint forall (m_i in 1..n_matchups) (
    forall (pos in 1..3) (
    matchup_table_team_elos[m_i,1,pos] = (ELO[matchup_table[m_i,POS[pos,1]]] + higher_rating_weight*ELO[matchup_table[m_i,POS[pos,2]]]) / (1+higher_rating_weight)
 /\ matchup_table_team_elos[m_i,2,pos] = (ELO[matchup_table[m_i,POS[pos,3]]] + higher_rating_weight*ELO[matchup_table[m_i,POS[pos,4]]]) / (1+higher_rating_weight)));


% argmin predicate i copied from http://hakank.org/minizinc/argmax.mzn
predicate argmin_le(var int: p, array[int] of var float: x) =
  forall(i in index_set(x)) ( x[p] <= x[i] )
; 

% Constraint that should help but somehow doesn't
% constraint (matchup_table[1,1] == 1); %% ASSUMES ELO IS SORTED DESC !!! % somehow leads to worse results


%function var float: reduce_if_below(var float: value, float: threshold, float: factor) =
%  if value <= threshold then value*factor else value-threshold + threshold*factor endif;

function var float: reduce_if_below(var float: value, float: threshold) =
  if value <= threshold then value*0.01 else value-threshold + threshold*0.01 endif;

function var float: square(var float: value) = value * value

constraint symmetry_breaking_constraint (forall (match_i in 1..n_matchups-1)
(     matchup_table[match_i,1] < matchup_table[match_i+1,1] ) );