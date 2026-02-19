import java.awt.AWTEvent;
import java.awt.Component;
import java.awt.Window;
import java.awt.event.KeyEvent;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.SwingUtilities;

public class MegaMek {
    
    /*************   Method 32   *************/
    //SNIPPET_STARTS
    public Victory.Result victory(IGame game, HashMap<String, Object> ctx) {
        boolean victory = false;
        VictoryResult vr = new VictoryResult(true);
        
        Hashtable<Integer,Integer> killsTeam = new Hashtable<Integer,Integer>();
        
        Hashtable<Integer,Integer> killsPlayer = new Hashtable<Integer,Integer>();
        
        updateKillTables(game, killsTeam, killsPlayer, game.getWreckedEntities());
        updateKillTables(game, killsTeam, killsPlayer, game.getCarcassEntities());
        
        boolean teamHasHighestKills = true;
        int highestKillsId = -1;
        int killCount = 0;
        for (Integer killer : killsTeam.keySet()){
            if (killsTeam.get(killer) > killCount){
                highestKillsId = killer;
                killCount = killsTeam.get(killer);
            }
        }
        
        for (Integer killer : killsPlayer.keySet()){
            if (killsTeam.get(killer) > killCount){
                highestKillsId = killer;
                killCount = killsPlayer.get(killer);
                teamHasHighestKills = false;
            }
        }
        
        if (killCount >= killCondition){
            Report r = new Report(7106, Report.PUBLIC);
            victory = true;
            if (teamHasHighestKills) {
                r.add("Team " + highestKillsId);
                vr.addTeamScore(highestKillsId, 1.0);                
            } else {
                IPlayer winner = game.getPlayer(highestKillsId);
                r.add(winner.getName());
                vr.addPlayerScore(winner.getId(), 1.0);
            }
            r.add(killCount);
            vr.addReport(r);
        }
        
        if (victory)
            return vr;
        return new SimpleNoResult();
    }
    
    /*************   Method 33   *************/ 
    //SNIPPET_STARTS
    private void parseAdvantages(Entity entity, String adv) {
        StringTokenizer st = new StringTokenizer(adv);

        while (st.hasMoreTokens()) {
            String curAdv = st.nextToken();
            int curParameter = 0;
            boolean bParameterDetected = false;

            StringTokenizer advantageParameterTokenizer = new StringTokenizer(
                    curAdv, ":");
            if (advantageParameterTokenizer.countTokens() > 1) {
                
                curAdv = advantageParameterTokenizer.nextToken();
                String curParam = advantageParameterTokenizer.nextToken();
                curParameter = Integer.parseInt(curParam);
                bParameterDetected = true;
            }

            IOption option = entity.getCrew().getOptions().getOption(curAdv);

            if (null == option) {
                System.out.println("Ignoring invalid pilot advantage: "
                        + curAdv);
            } else {
                System.out.println("Adding pilot advantage '" + curAdv
                        + "' to " + entity.getDisplayName());
                
                if (bParameterDetected) {
                    option.setValue(curParameter);
                } else {
                    option.setValue(true);
                }
            }
        }
    }
    
    /*************   Method 34   *************/
    //SNIPPET_STARTS
    private void roll(int connId, int dice, int sides) {
        StringBuffer diceBuffer = new StringBuffer();
        int total = 0;
        for (int i = 0; i < dice; i++) {
            int roll = Compute.randomInt(sides) + 1;
            total += roll;

            if (dice < 2) {
                diceBuffer.append(roll);
                continue;
            }
            
            if (i < dice - 1) {
                diceBuffer.append(roll);
                diceBuffer.append(", ");
            } else {
                diceBuffer.append("and ");
                diceBuffer.append(roll);
            }
        }
        server.sendServerChat(server.getPlayer(connId).getName()
                + " has rolled " + diceBuffer + " for a total of " + total
                + ", using " + dice + "d" + sides);
    }
    
    /*************   Method 35   *************/
    //SNIPPET_STARTS
    public void deploy(int id, Coords c, int nFacing, int elevation,
                    List<Entity> loadedUnits, boolean assaultDrop) {
        int packetCount = 6 + loadedUnits.size();
        int index = 0;
        Object[] data = new Object[packetCount];
        data[index++] = new Integer(id);
        data[index++] = c;
        data[index++] = new Integer(nFacing);
        data[index++] = new Integer(elevation);
        data[index++] = new Integer(loadedUnits.size());
        data[index++] = new Boolean(assaultDrop);

        for (Entity ent : loadedUnits) {
            data[index++] = new Integer(ent.getId());
        }

        send(new Packet(Packet.COMMAND_ENTITY_DEPLOY, data));
        flushConn();
    }
    
    /*************   Method 36   *************/
    //SNIPPET_STARTS
    public static boolean canMechFindClub(IGame game, int entityId) {
        final Entity entity = game.getEntity(entityId);
        if (null == entity.getPosition()) {
            return false;
        }
        final IHex hex = game.getBoard().getHex(entity.getPosition());

        if (!(entity instanceof BipedMech || entity instanceof TripodMech)) {
            return false;
        }

        if (entity.isShutDown() || !entity.getCrew().isActive()) {
            return false;
        }

        if (game.getOptions().booleanOption("no_clan_physical")
            && entity.isClan()) {
            return false;
        }

        if ((hex.terrainLevel(Terrains.WOODS) < 1)
            && (hex.terrainLevel(Terrains.JUNGLE) < 1)
            && (hex.terrainLevel(Terrains.RUBBLE) < Building.MEDIUM)
            && (hex.terrainLevel(Terrains.ARMS) < 1)
            && (hex.terrainLevel(Terrains.LEGS) < 1)) {
            return false;
        }

        if (!entity.hasWorkingSystem(Mech.ACTUATOR_SHOULDER, Mech.LOC_RARM)
            || !entity.hasWorkingSystem(Mech.ACTUATOR_SHOULDER,
                                        Mech.LOC_LARM)
            || (!entity.hasWorkingSystem(Mech.ACTUATOR_HAND, Mech.LOC_RARM) && !((Mech) entity)
                .hasClaw(Mech.LOC_RARM))
            || (!entity.hasWorkingSystem(Mech.ACTUATOR_HAND, Mech.LOC_LARM) && !((Mech) entity)
                .hasClaw(Mech.LOC_LARM))) {
            return false;
        }

        if (entity.hasQuirk(OptionsConstants.QUIRK_NEG_NO_ARMS)) {
            return false;
        }

        if (entity.getClubs().size() > 0) {
            return false;
        }

        return true;
    }
    
    /*************   Method 37   *************/
    //SNIPPET_STARTS
    private void checkReady() {

        for (Enumeration<IPlayer> i = game.getPlayers(); i.hasMoreElements(); ) {
            final IPlayer player = i.nextElement();
            if (!player.isGhost() && !player.isObserver() && !player.isDone()) {
                return;
            }
        }

        if (game.getNoOfInitiativeRerollRequests() > 0) {
            resetActivePlayersDone();
            game.rollInitAndResolveTies();

            determineTurnOrder(IGame.Phase.PHASE_INITIATIVE);
            clearReports();
            writeInitiativeReport(true);
            sendReport(true);
            return; 
        }

        if (!game.phaseHasTurns(game.getPhase())
            && ((game.getPhase() != IGame.Phase.PHASE_LOUNGE) || (game.getNoOfEntities() > 0))) {
            endCurrentPhase();
        }
    }
    
    /*************   Method 38   *************/
    //SNIPPET_STARTS
    public final boolean unload(Entity unit) {
        Entity trooper = game.getEntity(troopers);
        if ((trooper == null) || !trooper.equals(unit)) {
            
            return false;
        }
        troopers = Entity.NONE.getId(); // Changed to allow compilation (added getId())
        return true;
    }
    
    /*************   Method 39   *************/
    //SNIPPET_STARTS
    public int getEnemyInitialBV(IGame game, IPlayer player) {
        int ret = 0;
        for (Enumeration<IPlayer> f = game.getPlayers(); f.hasMoreElements();) {
            IPlayer other = f.nextElement();
            if (other.isObserver())
                continue;
            if (other.isEnemyOf(player)) {
                ret += other.getInitialBV();
            }
        }
        return ret;
    }
    
    /*************   Method 49   *************/
    //SNIPPET_STARTS
    public void eventDispatched(AWTEvent event) {
        Object source = event.getSource();
        if (event instanceof KeyEvent
                && source instanceof Component) {
            
            if ((SwingUtilities.windowForComponent((Component) source) == _window)) {
                ((KeyEvent) event).consume();
            }
        }
    }
    
    // @Override // Removed to allow compilation
    /*************   Method 52   *************/ 
    //SNIPPET_STARTS
    public void run(int connId, String[] args) {
        int kickArg = server.isPassworded() ? 2 : 1;

        if (!canRunRestrictedCommand(connId)) {
            server.sendServerChat(connId,
                    "Observers are restricted from kicking others.");
            return;
        }
        if (server.isPassworded()
                && (args.length < 3 || !server.isPassword(args[1]))) {
            server
                    .sendServerChat(connId,
                            "The password is incorrect.  Usage: /kick <password> [id#]");
        } else
            try {
                int kickedId = Integer.parseInt(args[kickArg]);

                if (kickedId == connId) {
                    server.sendServerChat("Don't be silly.");
                    return;
                }

                server.sendServerChat(server.getPlayer(connId).getName()
                        + " attempts to kick player #" + kickedId + " ("
                        + server.getPlayer(kickedId).getName() + ")...");
                
                server.send(kickedId, new Packet(Packet.COMMAND_CLOSE_CONNECTION));
                server.getConnection(kickedId).close();

            } catch (ArrayIndexOutOfBoundsException ex) {
                server
                        .sendServerChat("/kick : kick failed.  Type /who for a list of players with id #s.");
            } catch (NumberFormatException ex) {
                server
                        .sendServerChat("/kick : kick failed.  Type /who for a list of players with id #s.");
            } catch (NullPointerException ex) {
                server
                        .sendServerChat("/kick : kick failed.  Type /who for a list of players with id #s.");
            }
    }
    
    /*************   Method 53   *************/
    //SNIPPET_STARTS
    public float getWeightArmor() {
        return (float) getEntity().getLabArmorTonnage();
    }
    
    public float getWeightAllocatedArmor() {

        float armorWeight = 0;
        if (!getEntity().hasPatchworkArmor()) {
            armorWeight += armor[0].getWeightArmor(getTotalOArmor(),
                    getWeightCeilingArmor());
        } else {
            for (int i = 0; i < armor.length; i++) {
                int points = getEntity().getOArmor(i);
                if (getEntity().hasRearArmor(i) &&
                        (getEntity().getOArmor(i, true) > 0)) {
                    points += getEntity().getOArmor(i, true);
                }
                armorWeight += armor[i].getWeightArmor(points,
                        getWeightCeilingArmor());
            }
        }
        return armorWeight;
    }
    
    /*************   Method 54   *************/
    //SNIPPET_STARTS
    protected int calcAttackValue() {
        int av = 0;

        double damage = ((InfantryWeapon)wtype).getInfantryDamage();
        if((ae instanceof Infantry) && !(ae instanceof BattleArmor)) {
            damage = ((Infantry)ae).getDamagePerTrooper();
            av = (int) Math.round(damage * 0.6 * ((Infantry)ae).getShootingStrength());
        }
        if(bDirect) {
            av = Math.min(av+(toHit.getMoS()/3), av*2);
        }
        if(bGlancing) {
            av = (int) Math.floor(av / 2.0);
        }
        return av;
    }
    
    /*************   Method 55   *************/
    //SNIPPET_STARTS
    protected void setMekHitLocLog() {
        String name = store.getString(MEK_HIT_LOC_LOG);
        if (name.length() != 0) {
            try {
                mekHitLocLog = new PrintWriter(new BufferedWriter(
                        new FileWriter(name)));
                mekHitLocLog.println("Table\tSide\tRoll");
            } catch (Throwable thrown) {
                thrown.printStackTrace();
                mekHitLocLog = null;
            }
        }
    }
    
    /*************   Method 56   *************/
    //SNIPPET_STARTS
    protected double updateAVforAmmo(double current_av, AmmoType atype,
        WeaponType bayWType, int range, int wId) {

        Mounted mLinker = weapon.getLinkedBy();
        int bonus = 0;
        if ((mLinker != null && mLinker.getType() instanceof MiscType
                && !mLinker.isDestroyed() && !mLinker.isMissing()
                && !mLinker.isBreached() && mLinker.getType().hasFlag(
                MiscType.F_ARTEMIS))
                && atype.getMunitionType() == AmmoType.M_ARTEMIS_CAPABLE) {
            bonus = (int) Math.ceil(atype.getRackSize() / 5.0);
            if (atype.getAmmoType() == AmmoType.T_SRM) {
                bonus = 2;
            }
            current_av = current_av + bonus;
        }
        
        if (((mLinker != null) && (mLinker.getType() instanceof MiscType)
                && !mLinker.isDestroyed() && !mLinker.isMissing()
                && !mLinker.isBreached() && mLinker.getType().hasFlag(
                MiscType.F_ARTEMIS_V))
                && (atype.getMunitionType() == AmmoType.M_ARTEMIS_V_CAPABLE)) {
            
            bonus = (int) Math.ceil(atype.getRackSize() / 5.0);
            if (atype.getAmmoType() == AmmoType.T_SRM) {
                bonus = 2;
            }
        }

        if (atype.getMunitionType() == AmmoType.M_CLUSTER) {
            current_av = Math.floor(0.6 * current_av);
        } else if (AmmoType.T_ATM == atype.getAmmoType()) {
            if (atype.getMunitionType() == AmmoType.M_EXTENDED_RANGE) {
                current_av = bayWType.getShortAV() / 2;
            } else if (atype.getMunitionType() == AmmoType.M_HIGH_EXPLOSIVE) {
                current_av = 1.5 * current_av;
                if (range > WeaponType.RANGE_SHORT) {
                    current_av = 0.0;
                }
            }
        } else if (atype.getAmmoType() == AmmoType.T_MML
                && !atype.hasFlag(AmmoType.F_MML_LRM)) {
            current_av = 2 * current_av;
            if (range > WeaponType.RANGE_SHORT) {
                current_av = 0;
            }
        } else if (atype.getAmmoType() == AmmoType.T_AR10) {
            if (atype.hasFlag(AmmoType.F_AR10_KILLER_WHALE)) {
                current_av = 4;
            } else if (atype.hasFlag(AmmoType.F_AR10_WHITE_SHARK)) {
                current_av = 3;
            } else {
                current_av = 2;
            }
        }
        return current_av;
    }
    
    /*************   Method 57   *************/ 
    //SNIPPET_STARTS
    public boolean correctHeatSinks(StringBuffer buff) {
        if ((aero.getHeatType() != Aero.HEAT_SINGLE) 
                && (aero.getHeatType() != Aero.HEAT_DOUBLE)) {
            buff.append("Invalid heatsink type!  Valid types are "
                    + Aero.HEAT_SINGLE + " and " + Aero.HEAT_DOUBLE
                    + ".  Found " + aero.getHeatType() + ".");
        }
        
        if (aero.getEntityType() == Entity.ETYPE_CONV_FIGHTER){
            int maxWeapHeat = countHeatEnergyWeapons();
            int heatDissipation = 0;
            if (aero.getHeatType() == Aero.HEAT_DOUBLE){
                buff.append("Conventional fighters may only use single " +
                        "heatsinks!");
                return false;
            } 
            heatDissipation = aero.getHeatSinks();
            
            if(maxWeapHeat > heatDissipation) {
                buff.append("Conventional fighters must be able to " +
                        "dissipate all heat from energy weapons! \n" +
                        "Max energy heat: " + maxWeapHeat + 
                        ", max dissipation: " + heatDissipation);
                return false;
            } else {
                return true;
            }
        } else {
            return true;
        }        
    }
    //SNIPPETS_END
    

    //SNIPPET_STARTS
    protected void receivePlayerInfo(Packet c) {
        int pindex = c.getIntValue(0);
        IPlayer newPlayer = (IPlayer) c.getObject(1);
        if (getPlayer(newPlayer.getId()) == null) {
            game.addPlayer(pindex, newPlayer);
        } else {
            game.setPlayer(pindex, newPlayer);
        }

        PreferenceManager.getClientPreferences().setLastPlayerColor(
                newPlayer.getColorIndex());
        PreferenceManager.getClientPreferences().setLastPlayerCategory(
                newPlayer.getCamoCategory());
        PreferenceManager.getClientPreferences().setLastPlayerCamoName(
                newPlayer.getCamoFileName());
    }
    

    
    // STUBS ADDED BY NADEESHAN
    public interface IGame {
        java.util.List<IEntity> getWreckedEntities();
        java.util.List<IEntity> getCarcassEntities();
        IPlayer getPlayer(int id);
        Entity getEntity(int id);
        IBoard getBoard();
        IOptions getOptions();


        enum Phase {
            PHASE_INITIATIVE,
            PHASE_LOUNGE
        }

        Enumeration<IPlayer> getPlayers();
        int getNoOfInitiativeRerollRequests();
        void rollInitAndResolveTies();
        Phase getPhase();
        boolean phaseHasTurns(Phase phase);
        int getNoOfEntities();
        void addPlayer(int pindex, MegaMek.IPlayer newPlayer);
        void setPlayer(int pindex, MegaMek.IPlayer newPlayer);
        
    }

    public interface IBoard {
        IHex getHex(Object pos);
    }

    public interface IHex {
        int terrainLevel(Terrains t);
    }

    public interface IOptions {
        boolean booleanOption(String name);
    }
    public enum Terrains {
        WOODS, JUNGLE, RUBBLE, ARMS, LEGS
    }

    public static class Building {
        static final int MEDIUM = 1;
    }

    public interface IEntity {}

    public interface IPlayer {
        String getName();
        int getId();
        boolean isGhost();
        boolean isObserver();
        boolean isDone();
        
        boolean isEnemyOf(IPlayer other);
        int getInitialBV();
        int getColorIndex();
        int getCamoCategory();
        String getCamoFileName();
    }

    public static class Victory {
        static class Result {}
    }

    public static class VictoryResult extends Victory.Result {
        VictoryResult(boolean b) {}
        void addTeamScore(int id, double score) {}
        void addPlayerScore(int id, double score) {}
        void addReport(Report r) {}
    }

    public static class SimpleNoResult extends Victory.Result {}

    public static class Report {
        static final int PUBLIC = 0;
        Report(int code, int type) {}
        void add(Object o) {}
    }

    public int killCondition = 1;

    public void updateKillTables(IGame game, java.util.Hashtable<Integer,Integer> killsTeam,
                                java.util.Hashtable<Integer,Integer> killsPlayer,
                                java.util.List<IEntity> entities) {
    }

    public static class Entity {
        static final Entity NONE = new Entity();
        static final int ETYPE_CONV_FIGHTER = 0;
        int id = 1; 
        Crew getCrew() { 
            return new Crew(); 
        }
        String getDisplayName() { 
            return "Entity"; 
        }
        int getId() { 
            return 0; 
        }
        Object getPosition() { 
            return new Object(); 
        }
        boolean isShutDown() { 
            return false; 
        }
        boolean isClan() { 
            return false; 
        }

        boolean hasWorkingSystem(String system, int loc) { 
            return true; 
        }
        boolean hasQuirk(String quirk) { 
            return false; 
        }
        List<Object> getClubs() { 
            return new java.util.ArrayList<>(); 
        }
        
        public boolean equals(Entity obj) {
            return true;
        }

        // Entity(int id) { this.id = id; }

        Entity() {}
        double getLabArmorTonnage() { return 20.0; }
        boolean hasPatchworkArmor() { return false; }
        int getOArmor(int index) { return 5; }
        int getOArmor(int index, boolean rear) { return 2; }
        boolean hasRearArmor(int index) { return true; }
    }

    public static class Mech extends Entity {
        static final String ACTUATOR_SHOULDER = "ACTUATOR_SHOULDER";
        static final String ACTUATOR_HAND = "ACTUATOR_HAND";

        static final int LOC_RARM = 1;
        static final int LOC_LARM = 2;

        boolean hasClaw(int loc) { return false; }
    }

    public static class Crew {
        Options getOptions() { return new Options(); }
        boolean isActive() { return true; }
    }

    public static class OptionsConstants {
        static final String QUIRK_NEG_NO_ARMS = "QUIRK_NEG_NO_ARMS";
    }

    public static class Options {
        IOption getOption(String name) { return new Option(); }
    }

    public interface IOption {
        void setValue(int value);
        void setValue(boolean value);
    }

    public static class Option implements IOption {
        @Override
        public void setValue(int value) { /* stub */ }

        @Override
        public void setValue(boolean value) { /* stub */ }
    }
    
    public static class Compute {
        static int randomInt(int max) {
            return (int)(Math.random() * max);
        }
    }

    public static class Server {
        Player getPlayer(int connId) { return new Player(); }
        void sendServerChat(String msg) {}
        boolean isPassworded() { return false; }
        boolean isPassword(String p) { return true; }
        void sendServerChat(int connId, String msg) {}
        void send(int connId, Packet p) {}
        Connection getConnection(int connId) { return new Connection(); }
    }

    public static class Connection implements Closeable {
        public void close() {}
    }

    public static class Player {
        String getName() { return "Player"; }
    }

    public Server server = new Server();

    public static class Coords {}

    public static class Packet {
        static final int COMMAND_ENTITY_DEPLOY = 1;
        static final int COMMAND_CLOSE_CONNECTION = 0;
        Packet(int command, Object[] data) {}
        Packet(int command) {}
        int getIntValue(int index) { return 0; }
        Object getObject(int index) { return index; }
    }

    public void send(Packet p) {}
    public void flushConn() {}
    
    public static class BipedMech extends Mech {}
    public static class TripodMech extends Mech {}
    
    public IGame game = IGame.class.cast(new Object()); // dummy assignment
    public void resetActivePlayersDone() {
    }
    public void determineTurnOrder(IGame.Phase phase) {
    }
    public void clearReports() {
    }
    public void writeInitiativeReport(boolean b) {
    }
    public void sendReport(boolean b) {
    }
    public void endCurrentPhase() {
    }
    
    public int troopers = -1;
    public class Game {
        // public Entity getEntity(int id) {
        //     return new Entity(id); // just return a new Entity with that ID
        // }
        // public void addPlayer(int index, IPlayer player) {}
        // public void setPlayer(int index, IPlayer player) {}
        // public IPlayer getPlayer(int id) { return IPlayer.class.cast(new Object()); }
    }

    

    public Window _window = new Window(null) {}; // dummy assignment
    
    public boolean canRunRestrictedCommand(int connId) {
        return true;
    }

    public Armor[] armor = new Armor[6];
    public static class Armor {
        public float getWeightArmor(int points, float ceiling) {
            return points * 1.0f; // dummy calculation
        }
    }
    public Entity getEntity() {
        return new Entity();
    }

    public int getTotalOArmor() {
        return 10; // dummy value
    }

    public float getWeightCeilingArmor() {
        return 100.0f; // dummy ceiling
    }

    public Object wtype = new Object();
    public Object ae = new Object(); 
    public boolean bDirect;
    public boolean bGlancing;
    public MoSCalculator toHit = new MoSCalculator();
    public static class InfantryWeapon {
        double getInfantryDamage() { return 10.0; }
    }

    public static class Infantry {
        double getDamagePerTrooper() { return 5.0; }
        double getShootingStrength() { return 1.0; }
    }

    public static class BattleArmor extends Infantry {}

    public static class MoSCalculator {
        public int getMoS() { return 3; }
    }
    
    public static final String MEK_HIT_LOC_LOG = "mekHitLocLog";
    public Store store = new Store();
    public PrintWriter mekHitLocLog = new PrintWriter(System.out);
    public static class Store {
        public String getString(String key) {
            return "mekHitLocLog.txt";
        }
    }
    
    public Mounted weapon = new Mounted(); // the weapon mounted
    public static class Mounted {
        Type getType() { return new Type(); }
        boolean isDestroyed() { return false; }
        boolean isMissing() { return false; }
        boolean isBreached() { return false; }
        MegaMek.Mounted getLinkedBy() {
            return new Mounted();
        }
    }

    public static class Type {
        boolean hasFlag(int flag) { return false; }
    }

    public static class MiscType extends Type {
        static final int F_ARTEMIS = 1;
        static final int F_ARTEMIS_V = 2;
    }

    public static class AmmoType {
        static final int M_ARTEMIS_CAPABLE = 1;
        static final int M_ARTEMIS_V_CAPABLE = 2;
        static final int M_CLUSTER = 3;
        static final int M_EXTENDED_RANGE = 4;
        static final int M_HIGH_EXPLOSIVE = 5;

        static final int T_SRM = 1;
        static final int T_ATM = 2;
        static final int T_MML = 3;
        static final int T_AR10 = 4;

        static final int F_MML_LRM = 1;
        static final int F_AR10_KILLER_WHALE = 2;
        static final int F_AR10_WHITE_SHARK = 3;

        int getMunitionType() { return 0; }
        int getAmmoType() { return 0; }
        int getRackSize() { return 5; }
        boolean hasFlag(int f) { return false; }
    }

    public static class WeaponType {
        static final int RANGE_SHORT = 1;
        int getShortAV() { return 1; }
    }
    

    public Aero aero = new Aero();  // the field referenced in the method

    public static class Aero {
        static final int HEAT_SINGLE = 1;
        static final int HEAT_DOUBLE = 2;

        int getHeatType() { return HEAT_SINGLE; }
        int getEntityType() { return Entity.ETYPE_CONV_FIGHTER; }
        int getHeatSinks() { return 10; } // arbitrary
    }

    public int countHeatEnergyWeapons() { 
        return 0;
    }
    
    public static class PreferenceManager {
        static Preferences getClientPreferences() { return new Preferences(); }

        public static class Preferences {
            void setLastPlayerColor(int color) {}
            void setLastPlayerCategory(int category) {}
            void setLastPlayerCamoName(String name) {}
        }
    }
    
    public Object getPlayer(int id) {
        return id;
    }
}