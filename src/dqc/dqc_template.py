class DQCTemplate:
    def __call__(self, data):
        validation_report = self.validation_report(data)
        statistics_report = self.statistics_report(data)
        return self.render_report(validation_report = validation_report,
                                  statistics_report = statistics_report)
    
    def validation_report(self, data):
        """Validates data for correctness and consistency"""
        raise NotImplementedError()
    

    def statistics_report(self, data):
        """"Creates a report concerning data statistics"""
        raise NotImplementedError()
    
    def render_report(self, validation_report, statistics_report):
        """"Creates the whole report from both validation and statistics reports"""
        raise NotImplementedError()