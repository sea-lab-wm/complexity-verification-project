    //SNIPPET_STARTS
    public Object s37() {
        for(AbstractPluginProgramFormating config : mConfigs)
            if(config != null && config.isValid())
                list.add(new ProgramReceiveTarget(this, config.getName(), config.getId()));

        if(list.isEmpty())
            list.add(new ProgramReceiveTarget(this, DEFAULT_CONFIG.getName(), DEFAULT_CONFIG.getId()));

        return list.toArray(new ProgramReceiveTarget[list.size()]);
    }

    // Snippet s38                                                                      /*ORIGINALLY COMMENTED OUT, ALTERED METHOD*/
