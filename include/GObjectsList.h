#pragma once
#include <ostream>
#include "GObject.h"
#include "framework.h"
#include <memory>
#include <vector>

class GObjectsList :  public GObject {

	// This class represents a list of GObject items, allowing for operations such as adding, removing, and finding objects.
        
private:
    GObject*            m_first;              // first item of list
    GObject*            m_last;               // last item of list
    

protected:
    int                 m_count;              // number of items in list 
	int                 m_data_max;           // maximum number of items in list (not used in this implementation, but can be useful for future extensions)
    
    std::vector<std::unique_ptr<GObject>>   m_data;

    bool                m_free_mode;

public:
                        
	                    GObjectsList(void) : m_first(nullptr), m_last(nullptr), m_count(0), m_data_max(0), m_free_mode(false) {}
                        ~GObjectsList(void) {}

    explicit            GObjectsList(size_t max_size);
    int                 GetTypeID() const override { return defObjectsList; }
	//--- methods to create and manage elements
    const               GObject* GetElement(int index) const;
    //--- methods to access protected data
    GObject*            At(int index) const;
    GObject*            First(void) const { return m_first; }
    GObject*            Last(void) const { return m_last; }
    int                 Count(void) const { return m_count; }
    void                Add(GObject* node);
    void                Remove(GObject* node);
    void                Remove(int index);
    void                Clear(void);
    //--- methods for working with files
    virtual bool        Save(const int file_handle);
    virtual bool        Load(const int file_handle);
    //--- method of identifying the object
    virtual int         Type(void) const { return 0; } // Default type, can be overridden
    //--- method of comparing the objects
    virtual int         Compare(const GObject* node, const int mode = 0) const { return 0; } // Default comparison, can be overridden

	//--- additional methods
    GObject*            Find(int type) const;
    GObject*            FindByName(const char* name) const;
    GObject*            FindById(UUID id) const;
    void                Sort(int mode = 0);
    void                Reverse(void);
    void                Print(void) const;
	void                PrintToFile(const char* filename) const;
    void                PrintToFile(int file_handle) const;
    void                PrintToStream(std::ostream& os) const;
    
    /*void                PrintToStream(std::ostream& os, int mode) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext, bool showCousinSibling) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext, bool showCousinSibling, bool showCount) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext, bool showCousinSibling, bool showCount, const char* separator) const;
    void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext, bool showCousinSibling, bool showCount, const char* separator, const char* end) const;
	void                PrintToStream(std::ostream& os, int mode, const char* prefix, const char* suffix, int indent, bool showId, bool showType, bool showName, bool showData, bool showPrevNext, bool showCousinSibling, bool showCount, const char* separator, const char* end, int maxItems) const;
    */

   
};

