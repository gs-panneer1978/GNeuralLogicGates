#pragma once
#include "GObjectsList.h"
#include "constants.h"
#include "GNeuralConnection.h"


class GConnectionsList :  public GObjectsList {


public:
	// Pass the maximum size up to the base class constructor
						GConnectionsList() {};
						GConnectionsList(size_t max_size) : GObjectsList(max_size) {}
						~GConnectionsList(void) {}
	int					GetTypeID() const override { return defArrayConnects; }
	virtual bool		CreateElement(int const index);
	virtual void		IncreaseTotal() { m_count++; }
	virtual int			Type(void)  const { return defArrayConnects; }
	
	//--- methods for working with files
	bool		Save(const int file_handle) override {
		return GObjectsList::Save(file_handle);
	}
	bool		Load(const int file_handle) override {
		return GObjectsList::Load(file_handle);
	}
	
	//--- method of comparing the objects
	int			Compare(const GObject* node, const int mode = 0) const override {
		return GObjectsList::Compare(node, mode);
	}

};

